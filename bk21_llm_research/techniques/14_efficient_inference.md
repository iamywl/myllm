# Efficient Inference for Large Language Models (대규모 언어 모델의 효율적 추론)

## 1. 기법의 정의

효율적 추론(Efficient Inference)이란, 대규모 언어 모델(LLM)의 서빙(serving) 및 추론(inference) 과정에서 발생하는 지연시간(latency), 처리량(throughput), 메모리 사용량(memory footprint)의 병목을 체계적으로 해소하기 위한 시스템 및 알고리즘 수준의 최적화 기법 전체를 지칭하는 것이다. 이는 단순한 모델 압축(quantization, pruning)과는 구분되며, 추론 시점의 연산 스케줄링, 메모리 관리, 디코딩 전략, 하드웨어 커널 최적화, 분산 서빙 아키텍처를 포괄하는 것이다.

LLM 추론의 핵심적 특성은 **자기회귀적(autoregressive) 토큰 생성**에 있다. Transformer 기반 디코더 모델은 시퀀스 내 각 토큰을 순차적으로 생성하며, 각 디코딩 스텝에서 전체 모델 파라미터를 메모리에서 로드해야 하는 것이다. 이로 인해 추론은 본질적으로 **메모리 대역폭 제한(memory-bandwidth bound)** 워크로드가 되며, 이는 학습(training)이 연산 제한(compute-bound)인 것과 근본적으로 상이한 특성인 것이다.

---

## 2. 기존 기법의 한계와 효율적 추론 등장 배경

### 2.1 자기회귀 디코딩의 근본적 병목

Transformer 디코더의 추론은 두 단계로 구분되는 것이다:

1. **Prefill 단계**: 입력 프롬프트의 모든 토큰을 한 번에 처리하여 초기 KV cache를 구성한다. 이 단계는 **compute-bound**이다.
2. **Decode 단계**: 토큰을 한 개씩 순차 생성한다. 각 스텝에서 모델의 전체 가중치를 메모리에서 읽어야 하므로 **memory-bandwidth bound**이다.

디코드 단계에서 단일 토큰 생성의 산술 강도(arithmetic intensity)를 분석하면 다음과 같다:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs per token}}{\text{Bytes loaded}} \approx \frac{2P}{2P \cdot \text{bytes per param}} = \frac{1}{\text{bytes per param}}
$$

여기서 $P$는 모델 파라미터 수이다. FP16 모델의 경우 산술 강도는 약 $0.5$ FLOP/byte인 것이다. NVIDIA A100 GPU의 피크 연산 능력이 312 TFLOPS(FP16)이고 메모리 대역폭이 2 TB/s인 점을 고려하면, 연산-메모리 균형점(roofline)은 약 156 FLOP/byte인 것이다. 즉, 자기회귀 디코딩은 GPU 연산 능력의 **0.3%**만을 활용하는 극도로 비효율적인 연산인 것이다 (Pope et al., 2023).

### 2.2 메모리 용량 문제

LLM 추론에서 메모리는 세 요소에 의해 소비되는 것이다:

1. **모델 가중치**: LLaMA-2 70B의 경우 FP16 기준 약 140 GB
2. **KV Cache**: 시퀀스 길이와 배치 크기에 비례하여 증가
3. **활성화 메모리(Activation Memory)**: 중간 연산 결과

KV cache의 크기는 다음과 같이 산출되는 것이다:

$$
\text{KV Cache Size} = 2 \times B \times L \times H \times D_h \times S \times \text{dtype\_size}
$$

여기서 $B$는 배치 크기, $L$은 Transformer 레이어 수, $H$는 어텐션 헤드 수, $D_h$는 헤드 차원, $S$는 시퀀스 길이, dtype_size는 데이터 타입 크기(FP16의 경우 2 bytes)이다. 계수 2는 Key와 Value 두 텐서를 저장하기 때문인 것이다.

**구체적 예시**: LLaMA-2 70B ($L=80$, $H=64$, $D_h=128$)에서 시퀀스 길이 $S=4096$, 배치 크기 $B=1$인 경우:

$$
\text{KV Cache} = 2 \times 1 \times 80 \times 64 \times 128 \times 4096 \times 2 = 10.7 \text{ GB}
$$

배치 크기 $B=32$로 확장하면 약 **343 GB**가 필요하며, 이는 A100 80GB GPU 4장 이상의 메모리에 해당하는 것이다. 이 분석은 KV cache 관리가 효율적 추론의 핵심 과제임을 명확히 보여주는 것이다.

### 2.3 서빙 비용의 경제적 압박

GPT-4 규모 모델의 서빙에는 수천 개의 GPU가 필요하며, 추론 비용은 학습 비용을 수개월 내에 초과하는 것이다. Dettmers et al. (2023)의 분석에 따르면, LLaMA-65B 모델의 단일 요청 처리 비용은 A100 GPU 기준 약 $0.002-0.01이며, 일일 수백만 요청을 처리하는 서비스에서 연간 수천만 달러의 인프라 비용이 발생하는 것이다. 이러한 경제적 압박이 효율적 추론 연구의 실질적 동인(動因)인 것이다.

---

## 3. 주요 기법 상세

### 3.1 KV Cache: 기본 메커니즘과 한계

**문제**: 순수 자기회귀 디코딩에서 $t$번째 토큰 생성 시, 이전 $t-1$개 토큰의 Key, Value를 매번 재계산하면 연산량이 시퀀스 길이에 대해 $O(S^2)$으로 증가하는 것이다.

**해결**: 이전 스텝에서 계산된 Key, Value 텐서를 캐시에 저장하고 재사용한다. 이를 통해 각 디코딩 스텝의 연산량을 $O(S)$로 감소시키는 것이다.

```
Step t: query_t = W_q · x_t
        key_t = W_k · x_t,  value_t = W_v · x_t
        KV_cache = concat(KV_cache, (key_t, value_t))
        attn_t = softmax(query_t · KV_cache.keys^T / √d) · KV_cache.values
```

**새로운 문제**: KV cache는 연속적인 메모리 공간(contiguous memory)을 요구하며, 요청 간 시퀀스 길이의 가변성으로 인해 메모리 파편화(fragmentation)가 발생하는 것이다. 또한 최대 시퀀스 길이를 기준으로 사전 할당해야 하므로 실제 사용량 대비 60-80%의 메모리가 낭비되는 것이다 (Kwon et al., 2023).

### 3.2 Static Batching vs. Continuous Batching (Orca)

**문제**: 전통적인 정적 배칭(static batching)에서는 배치 내 모든 요청이 완료될 때까지 새로운 요청을 삽입할 수 없는 것이다. 요청의 출력 길이가 가변적이므로, 짧은 요청이 완료된 후에도 긴 요청의 완료를 대기하며 GPU가 유휴 상태가 되는 것이다.

**해결**: Yu et al. (2022)이 제안한 **Orca** 시스템은 **iteration-level scheduling**을 도입한 것이다. 각 디코딩 iteration마다 완료된 요청을 배치에서 제거하고 대기 중인 새 요청을 즉시 삽입하는 것이다. 이를 **continuous batching** (또는 **in-flight batching**)이라 하며, 핵심 설계는 다음과 같다:

1. **Selective Batching**: 요청별로 prefill/decode 단계를 독립적으로 스케줄링
2. **Iteration-level Scheduling**: 매 iteration마다 배치 구성을 동적 변경

**성능**: 정적 배칭 대비 처리량이 최대 **36.9배** 향상되었으며 (Yu et al., 2022), 이는 GPU 활용률의 극적인 개선을 의미하는 것이다.

**새로운 문제**: Continuous batching이 처리량을 극대화하더라도, 동적으로 변화하는 배치 크기에 따라 KV cache 메모리를 효율적으로 할당/해제하는 문제가 남는 것이다. 기존 메모리 관리 방식으로는 파편화가 불가피하며, 이것이 PagedAttention의 등장 배경인 것이다.

### 3.3 PagedAttention과 vLLM

**문제**: KV cache의 메모리 관리에서 세 가지 낭비가 발생하는 것이다:
- **내부 파편화(Internal Fragmentation)**: 최대 길이 기준 사전 할당 시 사용되지 않는 공간
- **외부 파편화(External Fragmentation)**: 가변 크기의 할당/해제로 인한 메모리 조각
- **예약 낭비(Reservation Waste)**: 미래 토큰을 위해 과도하게 예약된 메모리

**해결**: Kwon et al. (2023)이 제안한 **PagedAttention**은 운영체제의 가상 메모리(virtual memory) 및 페이징(paging) 기법을 KV cache 관리에 적용한 것이다.

KV cache를 고정 크기의 **블록(block)**으로 분할하며, 각 블록은 고정 개수의 토큰에 대한 Key, Value를 저장한다. 논리적 블록과 물리적 블록 간의 매핑은 **블록 테이블(block table)**을 통해 관리되므로, 물리적으로 비연속적인 메모리 공간에 KV cache를 저장할 수 있는 것이다.

메모리 효율을 정량적으로 분석하면:

$$
\text{Memory Waste}_{\text{paged}} = \frac{\text{Block Size} - 1}{2} \times \text{Bytes per Token}
$$

이는 마지막 블록에서만 평균적으로 (Block Size - 1)/2 토큰 분의 내부 파편화가 발생함을 의미하며, 외부 파편화는 **완전히 제거**되는 것이다. 블록 크기가 16 토큰일 때, 평균 낭비는 토큰당 약 7.5 토큰 분으로 전체 시퀀스 대비 무시할 수 있는 수준인 것이다.

**Copy-on-Write (CoW) 메커니즘**: Beam search나 parallel sampling에서 동일한 프롬프트의 KV cache를 공유하되, 분기(divergence)가 발생하는 시점에만 블록을 복사하는 것이다. 이를 통해 beam search에서 메모리 사용량을 최대 **55%** 절감한다.

**vLLM 시스템 성능**: PagedAttention을 기반으로 구축된 vLLM은 기존 HuggingFace Transformers 대비 처리량을 **14-24배**, FasterTransformer 대비 **2.2-3.5배** 향상시킨 것이다.

**새로운 문제**: PagedAttention은 메모리 효율을 해결했으나, 디코딩 자체의 순차적 병목(한 번에 1 토큰)은 여전히 존재하는 것이다. 이것이 speculative decoding 계열 연구의 동기가 되는 것이다.

### 3.4 Speculative Decoding

자기회귀 디코딩의 순차적 병목을 해소하기 위한 가장 체계적인 접근법인 것이다. Leviathan et al. (2023)과 Chen et al. (2023)이 독립적으로 동일한 아이디어를 제안하였다.

**핵심 원리**: 소형 **draft 모델** $M_q$가 $K$개의 토큰을 빠르게 생성하고, 대형 **target 모델** $M_p$가 이를 한 번의 forward pass로 병렬 검증하는 것이다. 검증은 **수정된 rejection sampling**을 기반으로 하며, 출력 분포가 target 모델의 분포와 **정확히 동일함이 수학적으로 보장**되는 것이다.

**알고리즘**:
1. Draft 모델이 순차적으로 $K$개의 토큰 $\tilde{x}_1, \tilde{x}_2, \ldots, \tilde{x}_K$를 생성 (각 토큰의 확률 분포 $q(\tilde{x}_i | x_{<i})$ 기록)
2. Target 모델이 한 번의 forward pass로 $K$개 위치에서의 확률 분포 $p(\cdot | x_{<i})$를 계산
3. 각 위치 $i$에서 수락 확률을 계산:

$$
P(\text{accept } \tilde{x}_i) = \min\left(1, \frac{p(\tilde{x}_i | x_{<i})}{q(\tilde{x}_i | x_{<i})}\right)
$$

4. 첫 번째 거절(rejection) 위치 $j$에서, 수정된 분포에서 대체 토큰을 샘플링:

$$
x_j \sim \text{norm}\left(\max\left(0, p(x | x_{<j}) - q(x | x_{<j})\right)\right)
$$

5. 모든 $K$개가 수락되면 추가로 1개 토큰을 target 모델에서 샘플링

**기대 수락 토큰 수**: Draft 모델과 target 모델의 분포 일치도를 $\alpha$라 하면, 한 번의 검증 라운드에서 기대 수락 토큰 수는:

$$
E[\text{accepted tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

$\alpha = 0.8$, $K = 5$일 때 약 4.0개의 토큰이 수락되며, 이는 target 모델 1회 호출로 평균 4개 토큰을 생성하는 것과 동등한 것이다.

**speedup 분석**: Target 모델 1회 호출 시간을 $T_p$, draft 모델 1회 호출 시간을 $T_q$라 하면:

$$
\text{Speedup} \approx \frac{E[\text{accepted}] \cdot T_p}{T_p + K \cdot T_q}
$$

$T_q \ll T_p$일 때 (예: draft 모델이 target의 1/10 크기), speedup은 $E[\text{accepted}]$에 근접하는 것이다. 실측에서 Leviathan et al. (2023)은 **2-3배 speedup**을 보고하였다.

**새로운 문제**: 별도의 draft 모델을 학습/유지해야 하며, draft 모델의 품질이 speedup을 결정하는 것이다. 또한 draft 모델의 vocabulary, tokenizer가 target 모델과 호환되어야 하는 제약이 존재하는 것이다.

### 3.5 Medusa: 단일 모델 다중 헤드 디코딩

**문제**: Speculative decoding은 별도의 draft 모델이 필요하며, draft와 target 모델 간의 호환성 문제가 있는 것이다.

**해결**: Cai et al. (2024)이 제안한 **Medusa**는 기존 LLM의 마지막 hidden state 위에 **다수의 경량 디코딩 헤드(decoding head)**를 추가하여, 각 헤드가 미래의 $k$번째 토큰을 동시에 예측하는 것이다.

$$
\hat{x}_{t+k} = \text{MedusaHead}_k(h_t), \quad k = 1, 2, \ldots, K
$$

여기서 $h_t$는 시점 $t$의 마지막 은닉 상태이다. 각 Medusa 헤드는 단일 linear layer 또는 2-layer MLP로 구성되며, 원래 모델 가중치를 동결(freeze)한 채 헤드만 미세조정하는 **Medusa-1**, 전체 모델과 함께 미세조정하는 **Medusa-2** 변형이 존재하는 것이다.

**Tree Attention**: 다수 헤드의 예측을 조합하여 복수의 후보 시퀀스를 tree 형태로 구성하고, 이를 단일 forward pass에서 병렬 검증하는 것이다. Top-$k$ 후보를 각 헤드에서 선택하면 $k^K$개의 후보 경로가 생성되나, **tree-based attention mask**를 활용하여 이를 효율적으로 처리하는 것이다.

**성능**: Vicuna-33B 모델에서 Medusa-1은 **2.2배**, Medusa-2는 **2.8배** speedup을 달성하였다.

**새로운 문제**: Medusa 헤드의 학습을 위한 추가 미세조정 비용이 발생하며, 헤드가 원본 모델의 은닉 상태에만 의존하므로 미래 토큰 간의 의존성(dependency)을 포착하지 못하는 것이다.

### 3.6 EAGLE: Feature 수준 자기회귀적 예측

**문제**: Medusa는 각 헤드가 독립적으로 미래 토큰을 예측하므로, 토큰 간 의존 관계를 모델링하지 못하는 것이다. 예를 들어 "New York City"에서 "York"의 예측은 "New"가 생성되었다는 정보에 의존해야 하나, Medusa는 이를 반영하지 못하는 것이다.

**해결**: Li et al. (2024a)이 제안한 **EAGLE**(Extrapolation Algorithm for Greater Language-model Efficiency)은 토큰 수준이 아닌 **feature 수준**에서 자기회귀적 예측을 수행하는 것이다. 구체적으로, 원본 LLM의 마지막 은닉 상태 $h_t$와 토큰 임베딩 $e_{t+1}$을 결합하여 다음 시점의 feature를 예측하는 경량 auto-regression head를 학습한다:

$$
\hat{h}_{t+1} = f_\theta(h_t, e_{t+1})
$$

이 예측된 feature $\hat{h}_{t+1}$을 다시 입력으로 사용하여 $\hat{h}_{t+2}$를 예측하는 방식으로 다단계 예측을 수행하는 것이다.

**성능**: EAGLE은 Vicuna-13B에서 **3.0배**, LLaMA-2 Chat 70B에서 **2.7배** speedup을 달성하며, 이는 Medusa 대비 약 **40-60%** 추가 향상인 것이다. lossless (출력 분포 동일) 방식 중 최고 수준의 speedup을 보고한 것이다.

### 3.7 EAGLE-2: Context-Aware Dynamic Draft Tree

**문제**: EAGLE-1을 포함한 기존 speculative decoding 방식은 고정된 tree 구조를 사용하며, 입력 컨텍스트에 따른 적응이 없는 것이다.

**해결**: Li et al. (2024b)이 제안한 **EAGLE-2**는 draft 모델의 신뢰도(confidence)에 기반하여 동적으로 draft tree를 구성하는 것이다. 높은 신뢰도의 경로는 더 깊게 탐색하고, 낮은 신뢰도의 경로는 조기에 가지치기(pruning)하는 **context-aware dynamic draft tree** 전략을 적용한다.

**성능**: EAGLE-2는 EAGLE-1 대비 추가적으로 **20-40%**의 speedup 향상을 달성하며, Vicuna-7B에서 **4.26배**, LLaMA-2 Chat 13B에서 **3.98배** speedup을 보고한 것이다.

### 3.8 Lookahead Decoding

**문제**: 기존 speculative decoding은 draft 모델의 학습이 필요하거나, 추가 파라미터가 요구되는 것이다.

**해결**: Fu et al. (2024)이 제안한 **Lookahead Decoding**은 **Jacobi iteration**에 기반한 병렬 디코딩 방식인 것이다. 핵심 통찰은, 자기회귀 디코딩을 비선형 방정식 시스템의 fixed-point iteration으로 재구성하는 것이다. 구체적으로, 미래 토큰 위치를 임의의 값으로 초기화한 후 반복적으로 정제(refinement)하며, 이 과정에서 생성되는 **n-gram pool**을 축적하여 검증 후보로 활용하는 것이다.

**장점**: 별도의 draft 모델이나 추가 학습이 불필요하며, 임의의 LLM에 즉시 적용 가능한 것이다.

**한계**: 실측 speedup이 speculative decoding 대비 낮으며 (약 1.5-2배), n-gram 기반이므로 반복적 패턴이 적은 텍스트에서는 효과가 제한적인 것이다.

### 3.9 Self-Speculative Decoding (Draft & Verify)

**문제**: Draft 모델을 별도로 학습/유지하는 오버헤드 없이 speculative decoding의 이점을 얻고자 하는 것이다.

**해결**: Zhang et al. (2023)이 제안한 **self-speculative decoding**은 단일 모델의 **일부 레이어만 사용**하여 draft를 생성하는 것이다. 예를 들어, 32-layer 모델에서 초기 8개 레이어 + 마지막 LM head만으로 빠르게 draft를 생성한 후, 전체 32개 레이어로 검증하는 방식이다. **Layer skipping** 또는 **early exit** 전략을 활용한다.

**장점**: 추가 모델 없이 기존 모델 내에서 draft-verify 파이프라인을 구현 가능한 것이다.

**새로운 문제**: 얕은 레이어의 예측 품질이 draft 모델 대비 열등할 수 있으며, 수락률이 낮으면 speedup이 제한적인 것이다.

### 3.10 SGLang과 RadixAttention

**문제**: 현대 LLM 애플리케이션은 단일 호출이 아닌 **다중 호출(multi-turn, multi-branch)** 패턴을 보이며, 동일한 시스템 프롬프트나 few-shot 예시가 반복적으로 사용되는 것이다. 기존 서빙 시스템은 각 요청을 독립적으로 처리하여 공통 prefix의 KV cache를 재활용하지 못하는 것이다.

**해결**: Zheng et al. (2024)이 제안한 **SGLang**은 **RadixAttention** 기법을 도입한 것이다. 이는 **radix tree** (기수 트리) 자료구조를 활용하여 모든 요청의 KV cache를 prefix 기반으로 관리하는 것이다. 공통 prefix를 공유하는 요청들은 동일한 KV cache 블록을 참조하며, LRU(Least Recently Used) 정책으로 캐시 교체를 관리한다.

추가적으로 SGLang은 다음 기능을 제공하는 것이다:
- **Constrained Decoding 최적화**: JSON, regex 등 출력 형식 제약 조건의 효율적 처리를 위한 FSM(finite state machine) 기반 compressed finite state machine
- **프론트엔드 DSL**: Python 기반 프로그래밍 인터페이스로 복잡한 LLM 프로그램의 표현

**성능**: 복잡한 LLM 프로그램(multi-turn chat, tree-of-thought, agent 워크플로우)에서 기존 시스템 대비 최대 **6.4배** 처리량 향상을 달성한 것이다.

### 3.11 TensorRT-LLM

**문제**: 범용 추론 프레임워크는 LLM 특화 최적화가 부족하며, GPU 하드웨어의 성능을 최대한 활용하지 못하는 것이다.

**해결**: NVIDIA의 **TensorRT-LLM**은 다음 최적화를 통합적으로 제공하는 것이다:
- **커스텀 CUDA 커널**: Flash Attention, fused MHA, fused GEMM 등 LLM 연산에 최적화된 커널
- **양자화 지원**: FP8, INT8 SmoothQuant, INT4 AWQ/GPTQ 등 다양한 양자화 방식
- **Tensor Parallelism / Pipeline Parallelism**: 다중 GPU 분산 추론
- **In-Flight Batching**: Continuous batching의 NVIDIA 구현
- **Paged KV Cache**: PagedAttention 통합

**성능**: 동일 하드웨어에서 HuggingFace 대비 최대 **8배** 이상의 처리량 향상을 제공하며, 특히 H100 GPU의 FP8 Tensor Core를 활용한 추론에서 최고 수준의 토큰/초 성능을 달성하는 것이다.

### 3.12 llama.cpp와 GGUF 포맷

**문제**: 대규모 GPU 클러스터가 아닌 소비자급 하드웨어(CPU, Apple Silicon, 저사양 GPU)에서 LLM 추론을 수행하고자 하는 요구가 증가한 것이다.

**해결**: Gerganov (2023)가 개발한 **llama.cpp**는 C/C++로 작성된 경량 LLM 추론 엔진으로, 다음 특징을 가지는 것이다:
- **GGUF 포맷**: 양자화된 모델 가중치를 단일 파일로 저장하는 효율적 파일 포맷
- **다양한 양자화**: Q2_K ~ Q8_0까지 2-8 bit 양자화 지원
- **CPU 최적화**: AVX2, AVX-512, ARM NEON 등 SIMD 명령어 활용
- **Apple Metal/CUDA 지원**: GPU 오프로딩을 통한 혼합 CPU-GPU 추론
- **메모리 매핑(mmap)**: 모델을 디스크에서 직접 메모리 매핑하여 로딩 시간 최소화

**의의**: LLM 추론의 민주화(democratization)에 핵심적으로 기여하였으며, Ollama, LM Studio 등 사용자 친화적 도구의 백엔드로 광범위하게 활용되는 것이다.

### 3.13 FlexGen: 단일 GPU 고처리량 추론

**문제**: 대형 LLM이 단일 GPU 메모리에 적재되지 않는 경우, 처리량을 최대화하면서 추론을 수행하고자 하는 것이다.

**해결**: Sheng et al. (2023)이 제안한 **FlexGen**은 **GPU, CPU, 디스크** 세 계층의 메모리를 통합적으로 활용하는 오프로딩(offloading) 기반 추론 엔진인 것이다. 핵심은 모델 가중치, KV cache, 활성화 값 각각을 어느 메모리 계층에 배치할지 최적화하는 **linear programming** 기반 탐색 알고리즘인 것이다.

$$
\text{Maximize: } \text{Throughput} = \frac{B \times S}{T_{\text{compute}} + T_{\text{transfer}}}
$$

여기서 $T_{\text{transfer}}$는 메모리 계층 간 데이터 전송 시간이다. FlexGen은 4-bit 양자화와 결합하여, 단일 NVIDIA T4 GPU (16GB)에서 OPT-175B 모델의 추론을 달성하는 것이다.

**한계**: 오프로딩 기반이므로 지연시간이 높으며, 대화형 서빙이 아닌 배치 처리(batch throughput) 시나리오에 적합한 것이다.

### 3.14 PowerInfer: 활성화 희소성 활용

**문제**: 소비자급 GPU에서 대형 LLM을 실시간으로 추론하고자 하나, 전체 모델 파라미터를 GPU에 적재할 수 없는 것이다.

**해결**: Song et al. (2023)이 제안한 **PowerInfer**는 LLM 추론 시 뉴런의 **활성화 패턴의 편향성(activation skewness)**을 활용하는 것이다. FFN(feed-forward network) 레이어에서 소수의 "hot" 뉴런이 대부분의 입력에 대해 활성화되는 반면, 다수의 "cold" 뉴런은 드물게 활성화되는 것이다.

- **Hot 뉴런** (전체의 ~10%): GPU에 상주하며 항상 연산
- **Cold 뉴런** (전체의 ~90%): CPU에 배치하며 필요시에만 연산
- **Adaptive Predictor**: 각 입력에 대해 어떤 cold 뉴런이 활성화될지 예측

**성능**: RTX 4090 단일 GPU에서 LLaMA-2 70B를 **11.69 tokens/s**로 추론하며, 이는 llama.cpp 대비 최대 **11.7배** 향상인 것이다.

**새로운 문제**: 활성화 예측기(predictor)의 정확도가 출력 품질에 직접 영향을 미치며, ReLU 기반이 아닌 활성화 함수(예: SwiGLU)에서는 희소성이 제한적인 것이다.

### 3.15 DistServe: Prefill/Decode 분리(Disaggregation)

**문제**: Prefill과 decode 단계는 근본적으로 상이한 연산 특성을 가지는 것이다. Prefill은 compute-bound이며 높은 GPU 활용률을 보이는 반면, decode는 memory-bandwidth bound이며 GPU 활용률이 극히 낮다. 이 두 단계를 동일 GPU에서 처리하면 **prefill-decode 간섭(interference)**이 발생하며, 특히 긴 프롬프트의 prefill이 진행 중인 decode 요청의 지연시간을 급격히 증가시키는 것이다 (일명 **stall problem**).

**해결**: Zhong et al. (2024)이 제안한 **DistServe**는 prefill과 decode를 **물리적으로 분리된 GPU 클러스터**에서 수행하는 **disaggregated serving** 아키텍처인 것이다.

- **Prefill GPU Pool**: Compute-intensive한 prefill 연산에 최적화
- **Decode GPU Pool**: Memory-bandwidth에 최적화, 배치 크기 극대화
- **KV Cache Transfer**: Prefill 완료 후 KV cache를 고속 네트워크(NVLink, InfiniBand)를 통해 decode 풀로 전송

**최적화 목표 (Goodput 최대화)**:

$$
\text{Goodput} = \frac{\text{SLO를 만족하는 요청 수}}{\text{전체 요청 수}}
$$

여기서 SLO(Service Level Objective)는 TTFT(Time to First Token)와 TPOT(Time Per Output Token)에 대한 지연시간 제약인 것이다. DistServe는 각 풀의 GPU 수와 병렬화 전략을 SLO 달성률을 최대화하도록 최적화한다.

**성능**: 동일 SLO 조건 하에서 기존 colocated 서빙 대비 **2-4배**의 goodput 향상을 달성한 것이다.

### 3.16 Splitwise

**문제**: DistServe와 유사한 문제 인식에서 출발하되, 대규모 프로덕션 환경에서의 비용 효율성에 초점을 맞춘 것이다.

**해결**: Patel et al. (2024)이 제안한 **Splitwise**는 Microsoft의 실제 프로덕션 LLM 서빙 데이터를 기반으로, prefill과 decode를 **이종 하드웨어(heterogeneous hardware)**에서 분리 실행하는 것이다. Prefill은 고연산 GPU에, decode는 고대역폭/저비용 GPU에 배치함으로써 **TCO(Total Cost of Ownership)**를 최적화한다.

**핵심 통찰**: Prefill과 decode를 분리함으로써, 각 단계에 최적화된 하드웨어를 독립적으로 스케일링할 수 있으며, 이는 동일 비용 대비 처리량을 크게 향상시키는 것이다.

### 3.17 Prefix Caching

**문제**: 다수의 요청이 동일한 시스템 프롬프트, few-shot 예시, 또는 공통 대화 이력(prefix)을 공유하는 것이다. 각 요청마다 이 공통 부분의 KV cache를 재계산하는 것은 중복 연산인 것이다.

**해결**: 공통 prefix의 KV cache를 캐시하여 후속 요청에서 재사용하는 것이다. 이는 두 가지 수준에서 구현되는 것이다:

1. **Prompt-level caching**: 정확히 동일한 프롬프트 prefix에 대해 KV cache를 재사용 (vLLM의 automatic prefix caching)
2. **Radix tree 기반**: SGLang의 RadixAttention이 대표적이며, 임의의 공통 prefix를 트리 구조로 관리

**효과**: 시스템 프롬프트가 긴 경우(예: 2048 토큰), TTFT를 **5-10배** 단축할 수 있으며, 총 연산량도 비례하여 감소하는 것이다.

### 3.18 KV Cache 압축 (Compression)

**문제**: 긴 시퀀스(32K, 128K 이상)에서 KV cache의 크기가 GPU 메모리를 초과하는 것이다.

**해결**: KV cache의 크기를 줄이기 위한 다양한 전략이 제안되었다:

1. **Eviction 기반**: H₂O (Heavy-Hitter Oracle, Zhang et al., 2024)는 어텐션 점수가 낮은 토큰의 KV를 동적으로 제거하는 것이다. 소수의 "heavy hitter" 토큰이 어텐션의 대부분을 차지한다는 관찰에 기반한다.
2. **Merging 기반**: 유사한 KV 벡터를 병합하여 캐시 크기를 축소
3. **Sparse Attention**: StreamingLLM (Xiao et al., 2024)은 초기 "attention sink" 토큰과 최근 토큰의 KV만 유지하는 **windowed attention with attention sink** 전략을 사용한다.

**StreamingLLM의 핵심 발견**: 초기 몇 개 토큰(attention sink)이 위치에 관계없이 높은 어텐션 점수를 받으며, 이를 제거하면 성능이 급격히 하락하는 것이다. 이 sink 토큰과 최근 윈도우 토큰만 유지함으로써, 이론적으로 **무한 길이** 시퀀스의 스트리밍 추론이 가능해진 것이다.

### 3.19 KV Cache 양자화 (Quantization)

**문제**: KV cache 압축의 가장 직접적인 방법으로, 정보 손실을 최소화하면서 KV cache의 비트 수를 줄이고자 하는 것이다.

**해결**: KV cache를 FP16에서 INT8 또는 FP8로 양자화하여 메모리를 50% 절감하는 것이다. Hooper et al. (2024)이 제안한 **KVQuant**는 다음 기법을 결합한다:

- **Per-channel quantization**: Key 텐서의 채널별 분포 차이를 반영
- **Per-token quantization**: Value 텐서의 토큰별 분포 차이를 반영
- **Non-uniform quantization**: K-means 기반 비균일 양자화로 분포 특성 보존
- **Dense-and-Sparse 분해**: Outlier 값을 별도로 저장하여 양자화 오류 최소화

**성능**: LLaMA-2 70B에서 KV cache를 **2-bit**까지 양자화하면서 FP16 대비 perplexity 증가를 **0.1 미만**으로 유지하는 것이다. 이를 통해 동일 메모리에서 최대 **8배** 더 긴 시퀀스 또는 더 큰 배치 크기를 처리할 수 있다.

### 3.20 Chunked Prefill

**문제**: 긴 프롬프트의 prefill이 진행되는 동안 decode 요청들이 차단(blocking)되어 TPOT이 급증하는 것이다. 이는 prefill이 compute-bound이며, 수십 밀리초에서 수 초까지 소요될 수 있기 때문이다.

**해결**: 긴 프롬프트의 prefill을 고정 크기의 **청크(chunk)**로 분할하고, 각 청크를 decode iteration 사이에 인터리빙(interleaving)하여 처리하는 것이다.

```
Iteration 1: [Prefill Chunk 1 (512 tokens)] + [Decode batch (32 requests)]
Iteration 2: [Prefill Chunk 2 (512 tokens)] + [Decode batch (32 requests)]
Iteration 3: [Prefill Chunk 3 (512 tokens)] + [Decode batch (32 requests)]
```

**효과**: Decode 요청의 TPOT 변동성을 크게 줄이며, P99 지연시간을 **50-80%** 개선하는 것이다. Agrawal et al. (2024)의 **Sarathi-Serve**가 이 기법을 체계화하였으며, 현재 vLLM, SGLang 등 주요 프레임워크에 통합되어 있다.

**새로운 문제**: 청크 크기의 선택이 prefill 효율과 decode 지연시간 간의 트레이드오프를 결정하며, 최적값은 워크로드 특성에 의존하는 것이다.

### 3.21 구조적 프루닝 (Structured Pruning)

**문제**: 모델 가중치 중 추론 품질에 기여하지 않는 부분을 제거하여 모델 크기와 연산량을 동시에 줄이고자 하는 것이다.

**해결**: LLM에 특화된 구조적 프루닝 기법이 다수 제안되었다:

- **LLM-Pruner** (Ma et al., 2023): Task-agnostic 구조적 프루닝으로, gradient 정보를 활용하여 중요도가 낮은 coupled structure(attention head, FFN 뉴런)를 식별하고 제거
- **Wanda** (Sun et al., 2024): 가중치 크기(magnitude)와 입력 활성화(activation)를 결합한 pruning metric으로, 학습 데이터 없이 단일 forward pass만으로 비구조적 프루닝을 수행
- **SparseGPT** (Frantar & Alistarh, 2023): 대규모 LLM에서 one-shot unstructured pruning을 수행하며, 50-60% 희소성에서 최소한의 성능 저하를 달성

**연산량 감소**:

$$
\text{FLOPs}_{\text{pruned}} = (1 - s) \times \text{FLOPs}_{\text{original}}
$$

여기서 $s$는 희소성 비율이다. 50% 비구조적 희소성의 경우 이론적으로 2배 speedup이나, 실제로는 하드웨어의 sparse 연산 지원 여부에 의존하는 것이다.

### 3.22 연산자 융합 (Operator Fusion)

**문제**: GPU에서 각 연산(GEMM, LayerNorm, activation, residual add 등)은 별도의 CUDA 커널로 실행되며, 커널 간 메모리 읽기/쓰기 및 커널 런치 오버헤드가 발생하는 것이다.

**해결**: 논리적으로 연속된 연산들을 단일 CUDA 커널로 융합(fuse)하여 중간 결과의 메모리 왕복을 제거하는 것이다. 주요 융합 패턴은 다음과 같다:

1. **QKV Projection Fusion**: Q, K, V 세 개의 linear projection을 단일 GEMM으로 결합
2. **Attention + Softmax + Value Projection Fusion**: Flash Attention (Dao et al., 2022; Dao, 2023)이 대표적
3. **FFN Fusion**: Gate projection + Up projection + Activation + Down projection을 하나의 커널로 결합
4. **Add + LayerNorm Fusion**: Residual connection과 layer normalization을 단일 커널로 처리

**Flash Attention의 기여**: Dao et al. (2022)이 제안한 Flash Attention은 attention 연산을 **IO-aware** 알고리즘으로 재설계하여, HBM(High Bandwidth Memory)과 SRAM(on-chip) 간의 데이터 이동을 최소화하는 것이다. 이는 tiling과 recomputation을 활용하여, 표준 attention 대비 **2-4배** 속도 향상과 $O(N)$ 메모리 사용량(기존 $O(N^2)$ 대비)을 달성하였다. Flash Attention 2 (Dao, 2023)는 이를 더욱 개선하여 A100에서 이론적 피크의 **72%** 활용률을 달성한 것이다.

---

## 4. 기법 진화의 인과적 흐름

효율적 추론 기법의 발전은 엄밀한 **인과적 연쇄(causal chain)**를 따르는 것이다. 각 기법은 이전 기법이 해결하지 못한 구체적 문제에 대한 응답으로 등장하였다.

### Phase 1: 기본 메모리 최적화 (2017-2022)

```
자기회귀 디코딩의 반복 연산 문제
  → KV Cache 도입 (시간 복잡도 O(S²) → O(S))
    → 새로운 문제: KV cache 메모리가 배치 크기/시퀀스 길이에 비례 증가
    → 새로운 문제: 정적 배칭의 GPU 활용률 저하
```

### Phase 2: 서빙 시스템 혁신 (2022-2023)

```
정적 배칭의 비효율
  → Orca/Continuous Batching (Yu et al., OSDI 2022)
    → 처리량 극대화, but KV cache 메모리 파편화 심화
      → PagedAttention/vLLM (Kwon et al., SOSP 2023)
        → 메모리 파편화 해결, but 디코딩 순차성 미해결
```

### Phase 3: 디코딩 병렬화 (2023-2024)

```
순차 디코딩 병목
  → Speculative Decoding (Leviathan et al., ICML 2023; Chen et al., 2023)
    → 2-3배 speedup, but 별도 draft 모델 필요
      → Medusa (Cai et al., ICML 2024): 다중 헤드로 draft 모델 불필요화
        → 한계: 토큰 간 의존성 무시
          → EAGLE (Li et al., ICML 2024): Feature-level autoregression
            → EAGLE-2: Dynamic tree로 추가 향상
      → Lookahead (Fu et al., 2024): 학습 불필요 Jacobi 기반
      → Self-Speculative (Zhang et al., 2023): 단일 모델 내 draft
```

### Phase 4: 시스템 아키텍처 분화 (2023-2024)

```
Prefill-decode 간섭 문제
  → DistServe (Zhong et al., OSDI 2024): 물리적 분리
  → Splitwise (Patel et al., 2024): 이종 하드웨어 분리
  → Sarathi-Serve (Agrawal et al., 2024): Chunked prefill로 소프트웨어적 분리

메모리 용량 한계 (긴 컨텍스트)
  → KV Cache 압축: H₂O, StreamingLLM
  → KV Cache 양자화: KVQuant, KIVI
  → Prefix Caching: SGLang RadixAttention

소비자 하드웨어 추론 요구
  → llama.cpp/GGUF: CPU/혼합 추론 최적화
  → FlexGen: GPU-CPU-Disk 오프로딩
  → PowerInfer: 활성화 희소성 기반 GPU-CPU 분배
```

### Phase 5: 통합과 수렴 (2024-현재)

현재의 추세는 개별 기법의 **통합(integration)**인 것이다. 최신 서빙 프레임워크(vLLM, SGLang, TensorRT-LLM)는 continuous batching, PagedAttention, prefix caching, chunked prefill, speculative decoding, KV cache 양자화, Flash Attention을 동시에 지원하는 것이다. 또한 disaggregated serving과 speculative decoding의 결합, 희소성과 양자화의 동시 적용 등 **교차 최적화(cross-optimization)**가 활발히 연구되고 있다.

이 진화의 핵심 패턴은 다음과 같이 요약되는 것이다:

| 병목 유형 | 대응 기법 | 잔여 문제 |
|-----------|-----------|-----------|
| 반복 연산 | KV Cache | 메모리 증가 |
| GPU 유휴 | Continuous Batching | 메모리 파편화 |
| 메모리 파편화 | PagedAttention | 순차 디코딩 |
| 순차 디코딩 | Speculative Decoding | Draft 모델 의존 |
| Draft 모델 의존 | Medusa/EAGLE/Self-Spec | 시스템 수준 최적화 |
| Prefill-decode 간섭 | Disaggregated Serving | 네트워크 오버헤드 |
| 메모리 용량 (긴 컨텍스트) | KV Cache 압축/양자화 | 정보 손실 |
| 하드웨어 접근성 | llama.cpp/PowerInfer | 성능 한계 |

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Efficient Memory Management for Large Language Model Serving with PagedAttention | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica | **SOSP 2023** | https://arxiv.org/abs/2309.06180 |
| 2 | Orca: A Distributed Serving System for Transformer-Based Generative Models | Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun | **OSDI 2022** | https://www.usenix.org/conference/osdi22/presentation/yu |
| 3 | Fast Inference from Transformers via Speculative Decoding | Yaniv Leviathan, Matan Kalman, Yossi Matias | **ICML 2023** | https://arxiv.org/abs/2211.17192 |
| 4 | Accelerating Large Language Model Decoding with Speculative Sampling | Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper | arXiv 2023 (DeepMind) | https://arxiv.org/abs/2302.01318 |
| 5 | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao | **ICML 2024** | https://arxiv.org/abs/2401.10774 |
| 6 | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang | **ICML 2024** | https://arxiv.org/abs/2401.15077 |
| 7 | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang | **EMNLP 2024** | https://arxiv.org/abs/2406.16858 |
| 8 | SGLang: Efficient Execution of Structured Language Model Programs | Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph Gonzalez, Clark Barrett, Ying Sheng | arXiv 2024 | https://arxiv.org/abs/2312.07104 |
| 9 | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU | Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang | **ICML 2023** | https://arxiv.org/abs/2303.06865 |
| 10 | Efficiently Scaling Transformer Inference | Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean | **MLSys 2023** | https://arxiv.org/abs/2211.05102 |
| 11 | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen | **SOSP 2024** | https://arxiv.org/abs/2312.12456 |
| 12 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang | **OSDI 2024** | https://arxiv.org/abs/2401.09670 |
| 13 | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini | **ISCA 2024** | https://arxiv.org/abs/2311.18677 |
| 14 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré | **NeurIPS 2022** | https://arxiv.org/abs/2205.14135 |
| 15 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Tri Dao | **ICLR 2024** | https://arxiv.org/abs/2307.08691 |
| 16 | H₂O: Heavy-Hitter Oracle: Efficient Generative Inference of Large Language Models with Heavy Hitters | Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen | **NeurIPS 2024** | https://arxiv.org/abs/2306.14048 |
| 17 | Efficient Streaming Language Models with Attention Sinks (StreamingLLM) | Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis | **ICLR 2024** | https://arxiv.org/abs/2309.17453 |
| 18 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami | arXiv 2024 | https://arxiv.org/abs/2401.18079 |
| 19 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu | **ICML 2024** | https://arxiv.org/abs/2402.02750 |
| 20 | LLM-Pruner: On the Structural Pruning of Large Language Models | Xinyin Ma, Gongfan Fang, Xinchao Wang | **NeurIPS 2023** | https://arxiv.org/abs/2305.11627 |
| 21 | SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot | Elias Frantar, Dan Alistarh | **ICML 2023** | https://arxiv.org/abs/2301.00774 |
| 22 | A Simple and Effective Pruning Approach for Large Language Models (Wanda) | Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter | **ICLR 2024** | https://arxiv.org/abs/2306.11695 |
| 23 | Lookahead Decoding: Breaking the Sequential Dependency of LLM Decoding with Lookahead and Verification | Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang | arXiv 2024 | https://arxiv.org/abs/2402.02057 |
| 24 | Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding | Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra | **ACL 2024** | https://arxiv.org/abs/2309.08168 |
| 25 | Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee | **OSDI 2024** | https://arxiv.org/abs/2308.16369 |
| 26 | LLM in a Flash: Efficient Large Language Model Inference with Limited Memory | Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C. Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar | arXiv 2023 (Apple) | https://arxiv.org/abs/2312.11514 |
| 27 | Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time | Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, Beidi Chen | **ICML 2023** | https://arxiv.org/abs/2310.17157 |
| 28 | GQA: Training Generalized Multi-Query Attention Transformers from Multi-Head Checkpoints | Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yinfei Yang, Siddhartha Mishra, Antoine Ranzato | **EMNLP 2023** | https://arxiv.org/abs/2305.13245 |
| 29 | Fast Transformer Decoding: One Write-Head is All You Need (Multi-Query Attention) | Noam Shazeer | arXiv 2019 | https://arxiv.org/abs/1911.02150 |
| 30 | Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads | Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan | arXiv 2024 | https://arxiv.org/abs/2401.11181 |
| 31 | The Llama 3 Herd of Models | Meta AI | arXiv 2024 | https://arxiv.org/abs/2407.21783 |
| 32 | SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification | Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia | **ASPLOS 2024** | https://arxiv.org/abs/2305.09781 |
| 33 | Sequence Parallelism: Long Sequence Training from System Perspective | Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, Hao Zhang | **ACL 2023** | https://arxiv.org/abs/2105.13120 |
| 34 | S-LoRA: Serving Thousands of Concurrent LoRA Adapters | Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica | **MLSys 2024** | https://arxiv.org/abs/2311.03285 |
| 35 | CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving | Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hoffmann, Ari Holtzman, Junchen Jiang | **SIGCOMM 2024** | https://arxiv.org/abs/2310.07240 |

---

*본 문서는 LLM 효율적 추론 기법에 대한 박사 수준의 참고 자료로, 2024년 하반기까지의 주요 연구를 포괄적으로 정리한 것이다. 각 기법의 인과적 관계와 수학적 기반을 중심으로 서술하였으며, 추론 시스템 설계 시 기법 간 트레이드오프를 이해하는 데 목적을 둔 것이다.*
