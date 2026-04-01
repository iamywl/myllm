# LLM 연산 효율화 및 전력 절감 최신 연구 동향 (2024-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, OSDI, ASPLOS, ISCA, HPCA, MLSys) 중심
> 총 참고논문: 45편

---

## 1. 개요

대규모 언어 모델(LLM)의 규모가 수십~수천억 파라미터로 확대됨에 따라, 추론(inference) 및 학습(training) 과정에서의 연산량과 에너지 소비가 심각한 공학적 과제로 부상하였다. GPT-4급 모델의 단일 추론 요청에 수 와트의 전력이 소모되며, 대규모 서비스 환경에서는 이러한 비용이 기하급수적으로 증가한다. 이에 따라 모델의 정확도를 유지하면서 연산량을 절감하고 에너지 효율을 높이는 연구가 탑 컨퍼런스에서 핵심 주제로 자리 잡고 있다.

본 문서에서는 LLM 효율화 연구를 (1) 효율적 추론, (2) 모델 압축, (3) 효율적 어텐션 메커니즘, (4) 에너지 효율, (5) 효율적 미세조정, (6) Mixture of Experts 효율화, (7) 시스템 수준 최적화의 7개 분야로 나누어 최신 연구를 정리한다.

---

## 2. 효율적 추론 (Efficient Inference)

### 2.1 기법 등장 배경

LLM의 자기회귀(autoregressive) 디코딩은 토큰 단위로 순차 생성하므로 본질적으로 느리다. 특히 모델 규모가 커질수록 단일 토큰 생성에 소요되는 시간이 증가하여, 실시간 서비스에서 심각한 지연(latency) 문제가 발생한다. 이를 해결하기 위해 투기적 디코딩(speculative decoding), 조기 종료(early exit), KV-캐시 최적화 등의 기법이 등장하였다.

### 2.2 투기적 디코딩 (Speculative Decoding)

투기적 디코딩은 소형 모델(draft model)이 먼저 여러 토큰을 예측한 후, 대형 모델(target model)이 이를 병렬로 검증하는 방식이다. 검증을 통과한 토큰은 그대로 사용하고, 거부된 토큰부터 재생성하여 출력 품질을 유지하면서 처리 속도를 높인다. 드래프트 토큰 $x_t$의 수용 확률은 다음과 같이 정의된다:

$$P_{\text{accept}}(x_t) = \min\left(1, \frac{p_{\text{target}}(x_t | x_{<t})}{p_{\text{draft}}(x_t | x_{<t})}\right)$$

여기서 $p_{\text{target}}$과 $p_{\text{draft}}$는 각각 타겟 모델과 드래프트 모델의 조건부 확률 분포이다. 이 수용-거부 방식은 최종 출력 분포가 타겟 모델의 분포와 정확히 일치함을 보장한다.

**SpecExec** (Svirschevski et al., NeurIPS 2024)은 대규모 병렬 투기적 디코딩 기법으로, 타겟 모델 1회 반복당 최대 20개 토큰을 생성한다. 4-bit 양자화와 RAM 오프로딩을 결합하여 소비자용 GPU에서 500억 파라미터 이상의 모델을 초당 4-6 토큰 속도로 추론할 수 있음을 보였다 [1].

**Cascade Speculative Drafting** (Chen et al., NeurIPS 2024)은 수직 캐스케이드(Vertical Cascade)와 수평 캐스케이드(Horizontal Cascade)를 도입하여 드래프트 모델의 자기회귀 생성을 제거하고, 드래프팅 단계의 시간 할당을 최적화하였다. 이를 통해 표준 투기적 디코딩 대비 추가적인 속도 향상을 달성하였다 [2].

**Speculative Streaming** (Bhendawade et al., NeurIPS 2024 ENLSP Workshop)은 보조 모델 없이 단일 모델에서 투기적 디코딩을 수행하는 기법이다. 미세조정 목표를 다음 토큰 예측에서 미래 n-gram 예측으로 변경하여, 1.8-3.1배의 속도 향상을 달성하면서 생성 품질을 유지한다 [3].

**CTC 기반 Speculative Decoding** (Huang et al., NeurIPS 2024)은 비자기회귀 드래프트 모델이 드래프트 토큰 간 상관관계를 무시하는 한계를 해결하기 위해 CTC(Connectionist Temporal Classification) 기반 드래프팅을 제안하여 수용률(acceptance rate)을 개선하였다 [4].

### 2.3 조기 종료 (Early Exit)

**LayerSkip** (Elhoushi et al., Meta, ACL 2024)은 조기 종료와 자기 투기적 디코딩을 결합한 기법이다. 앞쪽 레이어가 토큰을 드래프트하고 뒤쪽 레이어가 검증하는 구조로, 단일 모델 내에서 작동한다. 레이어 드롭아웃 스케줄을 적용하여 앞쪽 레이어의 예측력을 강화하고, 유연한 정확도-지연 트레이드오프를 가능하게 한다 [5].

### 2.4 KV-캐시 최적화

Transformer 기반 LLM은 이전 토큰의 Key-Value 벡터를 캐싱하여 반복 연산을 피하지만, 긴 시퀀스에서 KV-캐시의 메모리 사용량이 급격히 증가하는 문제가 있다. KV-캐시의 메모리 사용량은 다음과 같이 산출된다:

$$M_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \times b \times p$$

여기서 $n_{\text{layers}}$는 레이어 수, $n_{\text{heads}}$는 어텐션 헤드 수, $d_{\text{head}}$는 헤드 차원, $L$은 시퀀스 길이, $b$는 배치 크기, $p$는 정밀도(바이트)이다. 예컨대 Llama-2-70B(80 레이어, 64 헤드, $d_{\text{head}}$=128)에서 FP16 기준 시퀀스 길이 4K, 배치 크기 1일 때 KV-캐시는 약 10GB를 차지한다.

**KVQuant** (Hooper, Kim et al., NeurIPS 2024)는 채널별 Key 양자화, pre-RoPE Key 양자화, 비균일 양자화, Dense-and-Sparse 양자화를 결합하여 3-bit 정밀도에서 퍼플렉시티 저하를 0.1 미만으로 억제하였다. 8-GPU 시스템에서 최대 1000만 토큰 컨텍스트를 처리할 수 있음을 보였다 [6].

**KIVI** (Liu, Yuan et al., ICML 2024)는 Key 캐시는 채널별(per-channel), Value 캐시는 토큰별(per-token)로 양자화해야 한다는 비대칭 특성을 발견하였다. 결과적으로 2-bit KV-캐시 양자화를 달성하여 피크 메모리를 2.6배 절감하고, 배치 크기를 4배까지 확대하며, 처리량을 2.35-3.47배 향상시켰다 [7].

**ShadowKV** (Sun, Chang et al., CMU/ByteDance, ICML 2025 Spotlight)는 저랭크(low-rank) Key 캐시를 GPU에 저장하고 Value 캐시를 CPU에 오프로딩하여 KV-캐시 메모리 사용량을 6배 이상 절감하였다. Llama-3.1-8B 모델에서 122K 컨텍스트 길이 기준 A100 GPU에서 3.04배의 생성 처리량 향상을 달성하였다 [8].

---

## 3. 모델 압축 (Model Compression)

### 3.1 기법 등장 배경

LLM의 파라미터 수가 수십~수천억에 달함에 따라, 모델을 메모리에 적재하는 것 자체가 하드웨어 제약으로 작용한다. 양자화(quantization)는 부동소수점 가중치를 저비트로 표현하여 메모리와 연산량을 동시에 절감하고, 프루닝(pruning)은 불필요한 가중치를 제거하며, 지식 증류(knowledge distillation)는 대형 모델의 지식을 소형 모델로 전이하여 효율적 배포를 가능하게 한다. 균일 양자화(uniform quantization)의 기본 연산인 Round-to-Nearest(RTN)는 다음과 같이 정의된다:

$$Q(w) = s \cdot \text{clamp}\left(\left\lfloor \frac{w}{s} \right\rceil + z, \; 0, \; 2^b - 1\right), \quad s = \frac{w_{\max} - w_{\min}}{2^b - 1}$$

여기서 $w$는 원본 가중치, $s$는 스케일 팩터, $z$는 영점(zero-point), $b$는 비트폭이다. 양자화 오차는 $\|W - Q(W)\|_F$로 측정되며, 이를 최소화하는 것이 PTQ 기법의 핵심 목표이다.

### 3.2 양자화 (Quantization)

**OneBit** (Xu, Han et al., NeurIPS 2024)는 1-bit 모델 압축 프레임워크로, 행렬 분해(matrix decomposition) 기반의 초기화 방법을 제안하여 극단적 저비트 양자화가 실현 가능함을 입증하였다 [9].

**DuQuant** (Lin et al., NeurIPS 2024)는 LLM 가중치 분포에서 일반 이상치(Normal Outlier)와 대규모 이상치(Massive Outlier)를 이중 변환(dual transformation)을 통해 처리함으로써, 매우 낮은 비트폭에서의 사후 학습 양자화(PTQ) 성능을 향상시켰다 [10].

**QTIP** (Tseng, Sun et al., NeurIPS 2024 Spotlight)는 격자 부호 양자화(Trellis Coded Quantization, TCQ)를 활용하여 초고차원으로 확장 가능한 최초의 LLM PTQ 기법이다. 랜덤 Hadamard 변환을 통한 비간섭(incoherence) 처리로 가중치를 근사적으로 i.i.d. 가우시안 분포로 만들어 양자화 효율을 극대화하였다 [11].

**AQLM** (Egiazarian, Panferov et al., ICML 2024)은 가법 양자화(Additive Quantization)를 LLM에 일반화하여, 학습된 입력 적응형 양자화와 Transformer 블록 간 공동 코드북 최적화를 수행한다. 파라미터당 3-bit 미만에서 파레토 최적인 최초의 기법으로, 2-bit 영역에서의 성능을 크게 향상시켰다 [12].

**BiLLM** (Huang et al., ICML 2024)은 1.08-bit 가중치만으로 고정밀 추론을 달성한 최초의 사후 학습 양자화 기법이다. LLaMA2-70B에서 8.41 퍼플렉시티를 기록하여, 다양한 LLM 패밀리에 걸쳐 1-bit 양자화의 실용성을 입증하였다 [13].

**QBB** (Bulat, Ouali et al., NeurIPS 2024)는 양자화된 가중치를 이진 기저(binary bases)의 조합으로 표현하여 극도로 효율적인 행렬 곱셈을 가능하게 한다. 초저비트폭에서 경쟁력 있는 정확도를 달성하면서 하드웨어 친화적 이진 연산을 지원한다 [14].

### 3.3 프루닝 (Pruning)

**ALPS** (Meng et al., NeurIPS 2024)는 LLM의 원샷(one-shot) 비구조적 프루닝을 위한 개선된 최적화 기법을 제안하여, SparseGPT 및 Wanda 대비 높은 희소성(sparsity)에서 더 나은 정확도 유지를 달성하였다 [15].

**SoBP (Structured Optimal Brain Pruning)** (Wei, Lu et al., EMNLP 2024)는 재학습 없는 구조적 프루닝 기법으로, 전역 1차 정보(global first-order information)를 활용한 구조 선택, 그리디 정제(greedy refinement), 모듈별 재구성을 수행한다. 3개 LLM 패밀리의 14개 모델에서 8개 데이터셋에 걸쳐 SOTA를 달성하였다 [16].

**ESPACE** (Song et al., NeurIPS 2024)는 활성화 차원 축소(activation dimensionality reduction)를 가중치 양자화 및 프루닝의 보완적 압축 축으로 제안하여, 중간 활성화를 압축함으로써 최소한의 정확도 손실로 모델 크기를 절감한다 [17].

### 3.4 지식 증류 (Knowledge Distillation)

**DistiLLM** (Ko et al., ICML 2024)은 자기회귀 LLM을 위한 통합 지식 증류 프레임워크를 제안하였다. 통일된 목적 함수의 부재를 해결하고, GPT-2, OPT, OpenLLaMA 패밀리에서 생성 성능과 학습 속도 모두에서 기존 KD 기준선을 상회하였다 [18].

**The Mamba in the Llama** (Wang et al., NeurIPS 2024)는 대형 Transformer를 하이브리드 선형 RNN으로 증류하는 방법을 제시하였다. 어텐션 레이어 가중치를 재사용하여 학술 GPU 자원만으로 증류를 수행하며, Llama3-8B-Instruct에서 증류된 모델이 AlpacaEval 2에서 GPT-4에 필적하는 성능을 보였다 [19].

---

## 4. 효율적 어텐션 메커니즘 (Efficient Attention)

### 4.1 기법 등장 배경

표준 Self-Attention의 시간 및 공간 복잡도는 시퀀스 길이에 대해 $O(n^2)$이다. 컨텍스트 윈도우가 수만~수십만 토큰으로 확장됨에 따라, 어텐션 연산이 추론 비용의 지배적 요소가 되었다. FlashAttention이 IO-aware 알고리즘으로 메모리 효율적 어텐션을 실현한 이후, 이를 더욱 최적화하고 희소 어텐션을 적응적으로 학습하는 연구가 활발히 진행되고 있다. 표준 어텐션의 HBM(High Bandwidth Memory) 접근 횟수는 $O(N^2 d)$인 반면, FlashAttention은 타일링(tiling) 기법을 통해 이를 다음과 같이 절감한다:

$$\text{IO}_{\text{FlashAttention}} = O\left(\frac{N^2 d^2}{M}\right)$$

여기서 $N$은 시퀀스 길이, $d$는 헤드 차원, $M$은 SRAM 크기이다. $d^2 \ll M$ (일반적 GPU 조건)일 때 이는 표준 어텐션 대비 $O(d / M)$배의 IO 절감을 달성한다.

### 4.2 FlashAttention 확장

**FlashAttention-3** (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, NeurIPS 2024 Spotlight)는 Hopper GPU의 Tensor Core와 TMA(Tensor Memory Accelerator)의 비동기성을 워프 특화(warp-specialization)를 통해 활용한다. 블록 단위 행렬곱과 소프트맥스를 인터리빙하고, FP8 블록 양자화를 적용하여 FlashAttention-2 대비 1.5-2.0배 속도 향상을 달성하였다. H100에서 최대 840 TFLOPs/s(활용률 85%)를 기록하였다 [20].

**FlashMask** (Wang, Zeng et al., ICLR 2025)는 FlashAttention에 열 단위(column-wise) 마스크 표현을 도입하여, 전체 마스크 행렬을 물리적으로 생성하지 않고도 슬라이딩 윈도우, 문서 마스킹 등 복잡한 마스크 패턴을 선형 메모리 복잡도로 지원한다 [21].

**FlashInfer** (Ye, Chen, Lai et al., UW/CMU, MLSys 2025 Best Paper)는 KV-캐시 저장 이질성을 조합 가능한 블록-희소 형식과 JIT 컴파일 프레임워크로 해결한다. 프리필, 디코드, 투기적 디코딩 등 다양한 서빙 시나리오를 지원하는 통합 커스터마이징 가능 어텐션 엔진이다 [22].

### 4.3 희소 및 적응적 어텐션

**SageAttention** (Zhang, Wei et al., Tsinghua, ICLR 2025)은 어텐션 연산을 8-bit으로 양자화하여 FlashAttention-2 대비 2.1배, xformers 대비 2.7배의 속도 향상을 달성하였다. 재학습 없이 플러그 앤 플레이 방식으로 적용 가능하며, 언어/이미지/비디오 모델에서 무시할 수 있는 수준의 메트릭 손실만 발생한다 [23].

**SeerAttention** (Gao, Zeng et al., ICLR 2025)은 블록 수준 희소성 패턴을 학습하는 적응적 희소 어텐션 기법이다. 32K 컨텍스트에서 90% 희소성을 달성하면서 최소한의 퍼플렉시티 손실을 유지하며, FlashAttention-2 대비 5.67배의 속도 향상을 보였다 [24].

**kNN Attention** (Haris, ICLR 2025)은 kNN 기반 희소 어텐션에 대한 이론적 기초를 제공하며, Self-Attention을 소프트맥스 분포에 대한 기댓값으로 재정식화하고 kNN 인덱스와 지연 Gumbel 샘플링을 활용하여 증명 가능한 효율적 근사를 제시한다 [25].

### 4.4 장문맥 서빙 최적화

**LServe** (Yang, Guo et al., MIT/NVIDIA, MLSys 2025)는 로컬, 스트라이드, 블록-희소 등 서로 다른 희소 어텐션 패턴을 단일 서빙 프레임워크로 통합하여, 128K 토큰까지의 시퀀스에서 상당한 처리량 향상을 달성한다 [26].

**SampleAttention** (Zhu et al., MLSys 2025)은 런타임 통계에 기반하여 중요한 어텐션 항목을 샘플링하는 적응적 구조화 희소 어텐션을 제안한다. 거의 무손실 품질을 유지하면서 장문맥 추론의 어텐션 연산을 크게 절감한다 [27].

---

## 5. 에너지 효율적 LLM 학습 및 추론

### 5.1 기법 등장 배경

LLM 추론이 전체 생명주기 탄소 배출량의 과반을 차지한다는 실증 분석이 등장하면서, 에너지 효율은 단순한 비용 절감을 넘어 환경적 지속가능성의 관점에서 중요한 연구 주제가 되었다.

**DynamoLLM** (Stojkovic, Zhang, Goiri, Torrellas, Choukse, HPCA 2025 Best Paper Award)는 LLM 추론 클러스터의 성능 SLO와 에너지 소비를 동시에 최적화하는 프레임워크이다. 동적 전압/주파수 스케일링(DVFS)과 모델 배치 전략을 도입하여 지연 목표를 충족하면서 에너지를 크게 절감한다 [28].

**LLMCompass** (Zhang et al., ISCA 2024)는 다양한 하드웨어 설계에 걸쳐 LLM 추론 특성을 분석하고, 현재 아키텍처가 LLM 워크로드에 비효율적임을 입증한다. LLM 최적화 하드웨어를 위한 다목적 평가 및 설계 공간 탐색 도구를 제공한다 [29].

**Energy Considerations of LLM Inference** (Patel et al., ACL 2025)는 다양한 모델 크기, 양자화 수준, 하드웨어 구성에 걸쳐 LLM 추론의 에너지 소비에 대한 포괄적 실증 분석을 제공한다. 추론이 LLM 전체 생명주기 탄소 배출의 과반을 차지한다는 사실을 확인하였다 [30].

---

## 6. 효율적 미세조정 (Efficient Fine-Tuning)

### 6.1 기법 등장 배경

LLM의 전체 파라미터를 미세조정하는 것은 막대한 GPU 메모리와 연산량을 요구한다. LoRA(Low-Rank Adaptation)가 저랭크 행렬을 통해 파라미터 효율적 미세조정(PEFT)의 돌파구를 마련한 이후, LoRA의 한계(최적 학습률 설정, 풀 파인튜닝과의 성능 격차 등)를 극복하기 위한 다양한 변형 기법이 제안되고 있다. LoRA의 핵심 아이디어는 사전학습 가중치 $W_0 \in \mathbb{R}^{d \times k}$에 저랭크 분해된 업데이트를 가산하는 것이다:

$$W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)$$

여기서 $r$은 랭크 하이퍼파라미터로, 학습 가능 파라미터 수를 $d \times k$에서 $r \times (d + k)$로 대폭 축소한다. 예컨대 $d = k = 4096$, $r = 16$일 때 학습 파라미터는 전체의 약 0.78%에 불과하다.

**DoRA (Weight-Decomposed Low-Rank Adaptation)** (Liu, Wang et al., NVIDIA, ICML 2024 Oral)은 사전학습 가중치를 크기(magnitude)와 방향(direction) 성분으로 분해하고, LoRA를 방향 성분에만 적용한다. 동일 파라미터 예산에서 LoRA를 일관되게 상회하며, 풀 파인튜닝과의 격차를 좁혔다 [31].

**LoRA+** (Hayou, Ghosh, Yu, ICML 2024)는 LoRA 행렬 A와 B에 동일 학습률을 사용하는 것이 넓은 모델에서 차선(suboptimal)임을 증명하고, 고정 비율의 차별화된 학습률을 적용하여 최대 2배 빠른 미세조정과 1-2% 성능 향상을 동일 연산 비용으로 달성하였다 [32].

**GaLore (Gradient Low-Rank Projection)** (Zhao, Zhang et al., ICML 2024 Oral)은 학습 중 그래디언트를 저랭크 부분공간으로 투영하여, LoRA를 능가하는 메모리 효율로 전체 파라미터 학습을 가능하게 한다. 옵티마이저 메모리를 최대 82.5% 절감하여, 단일 24GB 소비자용 GPU에서 모델 병렬화나 오프로딩 없이 7B 모델의 사전학습을 가능하게 하였다 [33].

**HydraLoRA** (Tian et al., NeurIPS 2024)는 비대칭 LoRA 아키텍처로, 서로 다른 구성 요소에 서로 다른 랭크를 부여하여 파라미터 효율과 모델 품질 간의 트레이드오프를 해결한다 [34].

**VB-LoRA** (Li et al., NeurIPS 2024)는 레이어 간 공유 벡터 뱅크를 사용하여 LoRA 행렬을 구성함으로써 극단적 파라미터 효율을 달성한다. 표준 LoRA 대비 학습 가능 파라미터를 한 자릿수 이상 절감하면서 경쟁력 있는 성능을 유지한다 [35].

**Trans-LoRA** (Wang et al., NeurIPS 2024)는 원본 학습 데이터 없이 LoRA 어댑터를 서로 다른 기반 모델 간에 전이할 수 있게 하여, 기반 모델 업데이트 시의 모델 마이그레이션 문제를 해결한다 [36].

**Multi-task LoRA Serving** (Xu et al., NeurIPS 2024)는 공유된 양자화 기반 모델 위에서 서로 다른 태스크의 양자화된 LoRA 어댑터를 동시에 서빙하는 기법을 제안하여, 효율적 다중 태스크 추론을 가능하게 한다 [37].

---

## 7. Mixture of Experts (MoE) 효율화

### 7.1 기법 등장 배경

MoE 아키텍처는 모든 파라미터 중 일부 전문가(expert)만 활성화하여 연산량을 절감하지만, 표준 MoE는 약 8%의 전문가만 활성화되어 활용률이 낮고, 전문가 간 로드 밸런싱, 메모리 사용량, 통신 오버헤드 등의 문제가 존재한다.

**MoE++** (Yuan et al., ICLR 2025 Oral)는 영연산 전문가(zero-computation experts: zero, copy, constant)를 도입하여 단순한 토큰이 FFN 연산을 완전히 우회할 수 있게 한다. 전문가 순전파 처리량을 1.1-2.1배 향상시키면서, 도전적 토큰에 더 많은 FFN 전문가를 집중시켜 품질을 유지 또는 개선한다 [38].

**MH-MoE (Multi-Head Mixture-of-Experts)** (Wu, Huang et al., Microsoft, NeurIPS 2024)는 각 토큰을 여러 서브토큰으로 분할하여 다양한 전문가에 병렬 할당한 후 재통합하는 멀티헤드 메커니즘을 제안한다. 전문가 활용률을 개선하고 모델 품질을 향상시킨다 [39].

**Efficient MoE Inference** (Huang, Ardalani et al., Meta/UPenn, NeurIPS 2024)는 MoE 워크로드의 특성을 분석하고 배포 비효율의 원인을 규명한다. 동적 게이팅, 전문가 버퍼링, 전문가 로드 밸런싱을 제안하여 언어 모델 태스크에서 6.2-11.6배의 최대 처리량 향상을 달성하였다 [40].

**OLMoE** (Muennighoff et al., AI2, ICLR 2025 Oral)는 총 7B 파라미터이나 입력 토큰당 1B만 사용하는 완전 공개 MoE 언어 모델(OLMoE-1B-7B)이다. 가중치, 학습 데이터, 코드, 로그가 모두 공개되어 완전한 재현 가능 MoE 학습 레시피를 제공한다 [41].

---

## 8. 시스템 수준 LLM 서빙 최적화

### 8.1 기법 등장 배경

모델 수준의 효율화만으로는 대규모 서비스 환경의 요구사항을 충족하기 어렵다. 수천 명의 동시 사용자를 처리하면서 지연 SLO를 충족하고, 비용을 최소화하며, 하드웨어 장애에 대응하는 시스템 수준의 최적화가 필수적이다.

**ServerlessLLM** (Fu, Xue et al., OSDI 2024)는 최초의 서버리스 LLM 서빙 시스템으로, 다계층 체크포인트 로딩과 KV-캐시 재계산을 통한 라이브 마이그레이션을 특징으로 한다. 대규모 모델의 콜드 스타트 지연 문제를 해결한다 [42].

**Parrot** (Lin, Han et al., Microsoft, OSDI 2024)는 의미 변수(semantic variable)를 통합 추상화로 도입하여 서빙 시스템에 애플리케이션 수준 지식을 노출시킨다. 복합 AI 애플리케이션 내 다중 LLM 요청 간 데이터 흐름 분석을 가능하게 하여 교차 요청 최적화 기회를 제공한다 [43].

**SpotServe** (Miao et al., ASPLOS 2024)는 선점형 GPU 인스턴스에서의 최초의 분산 LLM 서빙 시스템이다. 병렬화 구성을 동적으로 적응시키고, 이분 그래프 매칭을 통한 최적 인스턴스 마이그레이션과 상태 보존 추론 복구를 도입하여, P99 지연을 2.4-9.1배 절감하고 비용을 54% 절약한다 [44].

**Efficient LLM Scheduling by Learning to Rank** (Fu et al., NeurIPS 2024)는 LLM 추론 요청의 스케줄링에 학습 기반 랭킹(learning-to-rank)을 적용하여, 입력 의존적 생성 길이 변동성을 반영한 최적 처리량과 지연을 달성한다 [45].

---

## 9. 종합 분석

### 9.1 압축과 추론 효율화의 수렴

양자화, 프루닝, 투기적 디코딩 등 개별적으로 발전해 온 효율화 기법들이 점차 결합되는 추세이다. SpecExec이 4-bit 양자화와 투기적 디코딩을 결합하고, KVQuant가 양자화와 KV-캐시 최적화를 통합한 사례에서 볼 수 있듯이, 단일 기법의 한계를 여러 기법의 직교적(orthogonal) 결합으로 극복하는 방향이 주류를 형성하고 있다.

### 9.2 시스템-알고리즘 공동 설계의 부상

FlashAttention-3, FlashInfer, LServe 등은 하드웨어 특성(Tensor Core 비동기성, HBM-SRAM 계층 구조)을 알고리즘 설계에 직접 반영한다. DynamoLLM과 SpotServe는 클러스터 수준의 자원 관리를 모델 서빙과 통합한다. 이는 모델 수준의 효율화만으로는 실제 배포 환경의 요구사항을 충족하기 어려우며, 시스템-알고리즘 공동 최적화가 필수적임을 시사한다.

### 9.3 에너지 효율의 독립적 연구 축 확립

LLM 추론의 탄소 배출이 전체 생명주기의 과반을 차지한다는 실증 결과가 보고되면서, 에너지 효율이 단순한 비용 절감을 넘어 독립적 연구 주제로 자리 잡았다. DVFS 기반 동적 전력 관리, 하드웨어 설계 공간 탐색 등 시스템 아키텍처 수준의 에너지 최적화가 탑 컨퍼런스에서 주목받고 있다.

### 9.4 PEFT의 심화와 다양화

LoRA 이후 DoRA, LoRA+, GaLore, VB-LoRA 등 다양한 변형이 등장하여, 파라미터 효율과 모델 품질 간의 파레토 프론티어를 지속적으로 확장하고 있다. 특히 어댑터의 모델 간 전이(Trans-LoRA), 다중 태스크 동시 서빙 등 실용적 배포 시나리오를 위한 연구가 활발히 진행되고 있다.

---

## 10. 참고문헌

[1] Svirschevski, R. et al. "SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices." NeurIPS 2024. https://arxiv.org/abs/2406.02532

[2] Chen, Z. et al. "Cascade Speculative Drafting for Even Faster LLM Inference." NeurIPS 2024. https://arxiv.org/abs/2312.11462

[3] Bhendawade, N. et al. "Speculative Streaming: Fast LLM Inference without Auxiliary Models." NeurIPS 2024 ENLSP Workshop. https://arxiv.org/abs/2402.11131

[4] Huang, Y. et al. "Speculative Decoding with CTC-based Draft Model for LLM Inference Acceleration." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/a79054a9da91d73ed3cb1a9e87d7cd2d-Abstract-Conference.html

[5] Elhoushi, M. et al. "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." ACL 2024. https://arxiv.org/abs/2404.16710

[6] Hooper, C., Kim, S. et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS 2024. https://arxiv.org/abs/2401.18079

[7] Liu, Z., Yuan, J. et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML 2024. https://arxiv.org/abs/2402.02750

[8] Sun, H., Chang, L.-W. et al. "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference." ICML 2025 (Spotlight). https://arxiv.org/abs/2410.21465

[9] Xu, Y., Han, X. et al. "OneBit: Towards Extremely Low-bit Large Language Models." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/7a7a3f53faafc0161be0fcb57e5fa078-Paper-Conference.pdf

[10] Lin, H. et al. "DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/9febda1c8344cc5f2d51713964864e93-Paper-Conference.pdf

[11] Tseng, A., Sun, Q. et al. "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024 (Spotlight). https://arxiv.org/abs/2406.11235

[12] Egiazarian, V., Panferov, A. et al. "AQLM: Extreme Compression of Large Language Models via Additive Quantization." ICML 2024. https://arxiv.org/abs/2401.06118

[13] Huang, W. et al. "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs." ICML 2024. https://arxiv.org/abs/2402.04291

[14] Bulat, A., Ouali, Y. et al. "QBB: Quantization with Binary Bases for LLMs." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/05b69cc4c8ff6e24c5de1ecd27223d37-Paper-Conference.pdf

[15] Meng, F. et al. "ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models." NeurIPS 2024. https://arxiv.org/abs/2406.07831

[16] Wei, J., Lu, Q. et al. "Structured Optimal Brain Pruning for Large Language Models (SoBP)." EMNLP 2024. https://aclanthology.org/2024.emnlp-main.775/

[17] Song, Y. et al. "ESPACE: Dimensionality Reduction of Activations for Model Compression." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/1f6591cc41be737e9ba4cc487ac8082d-Paper-Conference.pdf

[18] Ko, J. et al. "DistiLLM: Towards Streamlined Distillation for Large Language Models." ICML 2024. https://dl.acm.org/doi/10.5555/3692070.3693067

[19] Wang, J. et al. "The Mamba in the Llama: Distilling and Accelerating Hybrid Models." NeurIPS 2024. https://github.com/jxiw/MambaInLlama

[20] Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024 (Spotlight). https://arxiv.org/abs/2407.08608

[21] Wang, G., Zeng, J. et al. "FlashMask: Efficient and Rich Mask Extension of FlashAttention." ICLR 2025. https://arxiv.org/abs/2410.01359

[22] Ye, Z., Chen, L., Lai, R. et al. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." MLSys 2025 (Best Paper). https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf

[23] Zhang, J., Wei, J. et al. "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration." ICLR 2025. https://arxiv.org/abs/2410.02367

[24] Gao, Y., Zeng, Z. et al. "SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs." ICLR 2025. https://arxiv.org/abs/2410.13276

[25] Haris, T. "kNN Attention Demystified: A Theoretical Exploration for Scalable Transformers." ICLR 2025. https://openreview.net/forum?id=49v8meXjHS

[26] Yang, S., Guo, J. et al. "LServe: Efficient Long-Sequence LLM Serving with Unified Sparse Attention." MLSys 2025.

[27] Zhu, Q. et al. "SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention." MLSys 2025.

[28] Stojkovic, J., Zhang, C., Goiri, I., Torrellas, J., Choukse, E. "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency." HPCA 2025 (Best Paper Award). https://arxiv.org/abs/2408.00741

[29] Zhang, H. et al. "LLMCompass: Enabling Efficient Hardware Design for LLM Inference." ISCA 2024. https://augustning.com/assets/papers/llmcompass-isca-2024.pdf

[30] Patel, D. et al. "Energy Considerations of Large Language Model Inference and Efficiency Optimizations." ACL 2025. https://aclanthology.org/2025.acl-long.1563.pdf

[31] Liu, S.-Y., Wang, C.-Y. et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML 2024 (Oral). https://arxiv.org/abs/2402.09353

[32] Hayou, S., Ghosh, N., Yu, B. "LoRA+: Efficient Low Rank Adaptation of Large Models." ICML 2024. https://arxiv.org/abs/2402.12354

[33] Zhao, J., Zhang, Z. et al. "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection." ICML 2024 (Oral). https://arxiv.org/abs/2403.03507

[34] Tian, Z. et al. "HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/123fd8a56501194823c8e0dca00733df-Paper-Conference.pdf

[35] Li, Y., Han, S. et al. "VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/1e0d38c676d5855bcfab7f6d29d20ad9-Paper-Conference.pdf

[36] Wang, H. et al. "Trans-LoRA: Towards Data-free Transferable Parameter Efficient Finetuning." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/708fdc7911f11585ee7161518e509ae6-Paper-Conference.pdf

[37] Xu, H. et al. "Efficient Multi-task LLM Quantization and Serving for Multiple LoRA Adapters." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/747dc7c6566c74eb9a663bcd8d057c78-Paper-Conference.pdf

[38] Yuan, Z. et al. "MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts." ICLR 2025 (Oral). https://proceedings.iclr.cc/paper_files/paper/2025/hash/7efe88bb4138d602e56637cfcf713654-Abstract-Conference.html

[39] Wu, X., Huang, S. et al. "Multi-Head Mixture-of-Experts (MH-MoE)." NeurIPS 2024. https://arxiv.org/abs/2404.15045

[40] Huang, H., Ardalani, N. et al. "Toward Efficient Inference for Mixture of Experts." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html

[41] Muennighoff, N. et al. "OLMoE: Open Mixture-of-Experts Language Models." ICLR 2025 (Oral). https://arxiv.org/abs/2409.02060

[42] Fu, Y., Xue, L. et al. "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models." OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/fu

[43] Lin, C., Han, Z. et al. "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable." OSDI 2024. https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf

[44] Miao, X. et al. "SpotServe: Serving Generative Large Language Models on Preemptible Instances." ASPLOS 2024. https://arxiv.org/abs/2311.15566

[45] Fu, Y. et al. "Efficient LLM Scheduling by Learning to Rank." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf
