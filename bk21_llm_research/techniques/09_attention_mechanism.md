# Attention Mechanism & Transformer Architecture

## 1. 기법의 정의

Attention mechanism은 입력 시퀀스의 각 위치가 다른 모든 위치에 대해 가중 참조(weighted reference)를 수행함으로써, 고정 길이 병목(fixed-length bottleneck) 없이 가변 길이 문맥 정보를 동적으로 집계하는 신경망 연산이다. Transformer는 이 attention 연산을 유일한 시퀀스 모델링 수단으로 사용하여, 순환(recurrence) 및 합성곱(convolution) 구조를 완전히 대체한 아키텍처이다.

핵심 정의를 수식으로 표현하면 다음과 같다:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

여기서 Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{m×d_k}, V ∈ ℝ^{m×d_v}이며, n은 query 시퀀스 길이, m은 key-value 시퀀스 길이, d_k는 key 차원이다. 스케일링 팩터 √d_k는 내적 값의 분산을 1로 유지하여 softmax의 기울기 소실을 방지하는 역할을 수행한다.

---

## 2. 기존 기법의 한계와 Attention 등장 배경

### 2.1 Sequence-to-Sequence의 정보 병목 문제

**문제:** Sutskever et al. (2014)가 제안한 encoder-decoder 구조에서, encoder는 가변 길이 입력 시퀀스 전체를 하나의 고정 차원 벡터 c로 압축한다. 이 고정 벡터가 전체 소스 정보를 담아야 하므로, 시퀀스가 길어질수록 정보 손실이 불가피하다. 실험적으로 소스 문장 길이가 20 토큰을 초과하면 BLEU 점수가 급격히 하락하는 현상이 관측되었다.

**해결 (Bahdanau Attention):** Bahdanau et al. (2015)는 decoder의 각 타임스텝에서 encoder의 모든 hidden state에 대해 alignment score를 계산하고, 이를 가중합하여 context vector를 생성하는 방법을 제안하였다:

```
e_{ij} = a(s_{i-1}, h_j)                    # alignment model
α_{ij} = exp(e_{ij}) / Σ_k exp(e_{ik})      # attention weight
c_i = Σ_j α_{ij} h_j                        # context vector
```

여기서 s_{i-1}은 decoder의 이전 hidden state, h_j는 encoder의 j번째 hidden state, a(·)는 학습 가능한 alignment 함수(feedforward network)이다.

**새로운 문제:** 이 구조는 RNN 기반이므로, (1) encoder의 순차 계산으로 인한 병렬화 불가, (2) 역전파 경로가 긴 시퀀스에서 기울기 소실/폭발이 여전히 존재한다.

### 2.2 RNN/LSTM의 구조적 한계

**문제:** LSTM(Hochreiter & Schmidhuber, 1997)은 게이트 메커니즘으로 장거리 의존성을 완화하였으나, 본질적으로 시간 축을 따라 순차적으로 계산해야 한다. 이는 두 가지 근본적 한계를 야기한다:

1. **병렬화 불가:** 시간 t의 hidden state는 t-1에 의존하므로, GPU의 대규모 병렬 연산 능력을 활용할 수 없다.
2. **유효 문맥 길이 제한:** 이론적으로 무한 문맥을 처리할 수 있으나, 실제로는 수백 토큰 이후 정보가 크게 희석된다.

**해결 (Self-Attention):** 시퀀스 내 모든 위치 쌍 간의 직접적인 연결을 형성하면, 최대 경로 길이가 O(n)에서 O(1)로 감소한다. Vaswani et al. (2017)은 이 관찰을 기반으로 순환 구조 없이 self-attention만으로 시퀀스를 모델링하는 Transformer를 제안하였다.

**새로운 문제:** Self-attention의 계산 및 메모리 복잡도가 시퀀스 길이의 제곱 O(n²)이므로, 긴 시퀀스에 대한 확장성이 제한된다.

### 2.3 문제-해결의 인과적 요약

```
고정 벡터 병목 (Seq2Seq)
  → Bahdanau Attention (2015): encoder 전체를 동적 참조
    → 순차 계산 병목 (RNN 의존)
      → Self-Attention + Transformer (2017): 완전 병렬화
        → O(n²) 복잡도 병목
          → Flash Attention, Sparse Attention, Linear Attention 등
```

---

## 3. 주요 기법 상세

### 3.1 Scaled Dot-Product Attention

Vaswani et al. (2017)이 정의한 scaled dot-product attention의 수학적 형식은 다음과 같다:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**스케일링의 필요성에 대한 분석:** Q와 K의 각 원소가 평균 0, 분산 1인 독립 확률변수라고 가정하면, 내적 q·k = Σ_{i=1}^{d_k} q_i k_i의 분산은 d_k이다. d_k가 클 경우(예: 64 또는 128) 내적 값이 매우 커져 softmax의 출력이 one-hot에 가까워지고, 기울기가 극도로 작아진다. √d_k로 나누면 분산이 1로 정규화되어 softmax가 적절한 기울기를 유지한다.

**복잡도 분석:**
- 시간 복잡도: O(n²d) (n: 시퀀스 길이, d: 차원)
- 공간 복잡도: O(n² + nd) (attention 행렬 저장)

### 3.2 Multi-Head Attention (MHA)

단일 attention 함수는 하나의 표현 부분공간에서만 정보를 집계한다. Multi-head attention은 서로 다른 부분공간에서 독립적으로 attention을 수행하여 다양한 관계 패턴을 포착한다:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

여기서 W_i^Q ∈ ℝ^{d_model × d_k}, W_i^K ∈ ℝ^{d_model × d_k}, W_i^V ∈ ℝ^{d_model × d_v}, W^O ∈ ℝ^{hd_v × d_model}이다. 일반적으로 d_k = d_v = d_model / h로 설정하여, multi-head의 총 연산량이 단일 full-dimension attention과 동일하도록 한다.

**KV Cache 문제:** 자기회귀(autoregressive) 생성 시, 이전 토큰의 K, V를 재계산하지 않기 위해 캐시에 저장한다. MHA의 KV cache 크기는 다음과 같다:

```
KV Cache = 2 × n_layers × n_heads × d_head × seq_len × precision_bytes
```

예를 들어, LLaMA-70B (80 layers, 64 heads, d_head=128, fp16)에서 seq_len=4096이면 KV cache만 약 40GB가 필요하다. 이것이 MQA/GQA/MLA 연구의 직접적 동기이다.

### 3.3 Multi-Query Attention (MQA)

**문제:** MHA에서 KV cache는 추론 시 메모리의 지배적 요인이며, 배치 크기 확장의 병목이다.

**해결:** Shazeer (2019)는 모든 query head가 단일 K, V head를 공유하는 Multi-Query Attention을 제안하였다:

```
MQA: Q → h개 헤드, K → 1개 헤드, V → 1개 헤드
KV Cache 감소: 1/h (예: h=32이면 1/32)
```

**새로운 문제:** 표현력 감소로 인해 학습 품질이 MHA 대비 소폭(~0.5-1% perplexity) 하락하며, 특히 대규모 모델에서 이 차이가 누적될 수 있다.

### 3.4 Grouped-Query Attention (GQA)

**문제:** MQA는 KV cache를 극단적으로 줄이지만, 단일 KV head로는 표현력이 부족하다. MHA와 MQA 사이의 최적 지점이 존재할 수 있다.

**해결:** Ainslie et al. (2023)는 query head를 g개의 그룹으로 나누고, 각 그룹이 하나의 KV head를 공유하는 Grouped-Query Attention을 제안하였다:

```
GQA-g: h개 Q 헤드를 g개 그룹으로 분할
  - g = h: MHA와 동일
  - g = 1: MQA와 동일
  - 1 < g < h: MHA와 MQA의 절충

실제 구성 예시:
  LLaMA-2 70B: 64 Q헤드, 8 KV헤드 (g=8)
  LLaMA-3 8B:  32 Q헤드, 8 KV헤드 (g=8)
  Mistral 7B:  32 Q헤드, 8 KV헤드 (g=8)
```

핵심 기여는 기존 MHA 체크포인트를 GQA로 변환하는 "uptraining" 기법이다. 원래 MHA의 KV head를 그룹별 평균으로 초기화한 뒤, 소량의 추가 학습(원래 학습량의 5%)만으로 GQA의 성능을 MHA에 근접시킬 수 있다.

**새로운 문제:** GQA는 KV cache를 g배 줄이지만, 매우 긴 시퀀스(128K 이상)에서는 여전히 KV cache가 상당한 메모리를 차지한다.

### 3.5 Multi-head Latent Attention (MLA)

**문제:** GQA에서도 KV cache 크기는 seq_len에 선형 비례하여, 초장문 맥락에서 메모리 병목이 지속된다.

**해결:** DeepSeek-V2 (DeepSeek-AI, 2024)에서 도입된 MLA는 KV를 저차원 잠재 벡터(latent vector)로 압축하여 캐시한다:

```
# 압축 (학습 시)
c_t = W_DKV h_t                 # h_t ∈ ℝ^{d_model} → c_t ∈ ℝ^{d_c}, d_c << d_model

# 복원 (추론 시)
K_t = W_UK c_t                  # c_t에서 K 복원
V_t = W_UV c_t                  # c_t에서 V 복원

# Cache 대상: c_t만 저장 (K, V가 아님)
```

여기서 d_c는 잠재 차원으로, 일반적으로 d_model의 1/8~1/4 수준이다. DeepSeek-V2에서 d_c = 512, d_model = 5120으로 설정하여 KV cache를 GQA 대비 93.3% 감소시켰다.

**새로운 문제:** 복원을 위한 추가 행렬 곱셈이 필요하므로 연산량이 소폭 증가하며, RoPE와의 호환성을 위해 일부 head 차원에 별도의 위치 인코딩 경로가 필요하다.

### 3.6 위치 인코딩(Position Encoding)

#### 3.6.1 문제 정의

Self-attention 연산은 입력 토큰의 순서에 대한 정보를 전혀 포함하지 않는다. 수학적으로, 입력 시퀀스의 임의 치환(permutation)에 대해 attention 출력이 동일한 치환을 따르는, 즉 equivariant 성질을 가진다. 따라서 시퀀스의 순서 정보를 별도로 주입해야 한다.

#### 3.6.2 Sinusoidal Position Encoding

Vaswani et al. (2017)이 제안한 원래 방식으로, 고정된 삼각함수 패턴을 사용한다:

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

이 인코딩의 핵심 성질은 임의의 고정 오프셋 k에 대해 PE(pos+k)가 PE(pos)의 선형 변환으로 표현 가능하다는 것이다. 이론적으로 학습 시 본 적 없는 위치로의 외삽(extrapolation)이 가능하나, 실제로는 학습 길이를 크게 초과하면 성능이 저하된다.

#### 3.6.3 Learned Position Embedding

GPT-2 (Radford et al., 2019) 등에서 사용한 방식으로, 각 위치에 대해 학습 가능한 벡터를 할당한다. 구현이 단순하지만, 학습 시 설정한 최대 길이를 초과하는 위치에 대해서는 정의 자체가 존재하지 않아 외삽이 원천적으로 불가능하다.

#### 3.6.4 Rotary Position Embedding (RoPE)

**문제:** 절대 위치 인코딩은 외삽 성능이 불안정하고, 상대 위치 정보를 직접 인코딩하지 않는다.

**해결:** Su et al. (2021)이 제안한 RoPE는 query와 key 벡터에 위치에 따른 회전 변환(rotation)을 적용하여, 내적에서 자연스럽게 상대 위치 정보가 나타나도록 설계되었다.

2차원 부분공간에서의 회전 행렬:

```
R(θ_i, m) = [cos(mθ_i)  -sin(mθ_i)]
             [sin(mθ_i)   cos(mθ_i)]

여기서 θ_i = 10000^{-2i/d}, m은 토큰 위치
```

전체 d차원에 대한 RoPE:

```
f_q(x_m, m) = R_Θ,m · (W_Q x_m)
f_k(x_n, n) = R_Θ,n · (W_K x_n)

여기서 R_Θ,m = diag(R(θ_1, m), R(θ_2, m), ..., R(θ_{d/2}, m))
```

핵심 성질은 다음과 같다:

```
f_q(x_m, m)^T f_k(x_n, n) = (W_Q x_m)^T R_Θ,m^T R_Θ,n (W_K x_n)
                            = (W_Q x_m)^T R_Θ,(n-m) (W_K x_n)
```

즉, query와 key의 내적이 오직 상대 위치 (n-m)에만 의존하게 되어, 절대 위치 정보 없이 상대 위치 관계를 인코딩한다.

**새로운 문제:** RoPE는 학습 시 관측한 위치 범위를 초과하면 고주파 성분이 과도하게 회전하여 성능이 급락한다 (외삽 실패).

#### 3.6.5 ALiBi (Attention with Linear Biases)

**문제:** RoPE를 포함한 기존 위치 인코딩은 학습 길이를 초과하는 외삽에서 성능이 불안정하다.

**해결:** Press et al. (2022)는 위치 인코딩을 입력에 추가하는 대신, attention score에 거리 비례 편향을 직접 더하는 방식을 제안하였다:

```
ALiBi: softmax(q_i · k_j / √d_k - m · |i - j|)

여기서 m은 head별 기울기: m_h = 2^{-8h/H} (H: 총 head 수)
  예) H=8이면: m = {1/2, 1/4, 1/8, ..., 1/256}
```

각 head가 서로 다른 기울기를 가지므로, 일부 head는 국소적 패턴에, 다른 head는 장거리 패턴에 특화된다. 학습 시 본 적 없는 긴 위치에서도 편향이 자연스럽게 확장되므로, 외삽 성능이 RoPE 대비 우수하다.

**새로운 문제:** ALiBi의 선형 감쇠는 모든 종류의 장거리 의존성에 최적이 아니며, 특히 먼 위치의 중요 정보가 과도하게 억제될 수 있다. 실제로 대부분의 최신 LLM은 RoPE를 채택하고, 외삽 문제는 별도의 보간(interpolation) 기법으로 해결하는 경향이다.

#### 3.6.6 RoPE 외삽 기법: NTK-aware Interpolation과 YaRN

**문제:** RoPE 기반 모델의 문맥 길이를 학습 길이(예: 4K)에서 더 긴 길이(예: 128K)로 확장해야 한다.

**Position Interpolation (Chen et al., 2023):** 가장 단순한 접근으로, 위치 인덱스를 축소하여 학습 범위 내로 매핑한다:

```
m' = m × (L_train / L_target)
```

그러나 이 방식은 모든 주파수 대역을 균일하게 압축하여, 고주파 성분의 해상도가 불필요하게 손실된다.

**NTK-aware Interpolation (bloc97, 2023):** Neural Tangent Kernel 이론에 기반하여, 저주파는 보간하고 고주파는 외삽하는 적응적 방식이다:

```
θ_i' = θ_i / α^{2i/(d-2)}

여기서 α = (L_target / L_train)^{d/(d-2)}
```

이 변환은 RoPE의 base frequency를 변경하는 것과 동등하며, 저차원(저주파) 성분은 보간으로 안정적으로 처리하고, 고차원(고주파) 성분은 기존 해상도를 유지한다.

**YaRN (Peng et al., 2024):** Yet another RoPE extensioN은 NTK-aware interpolation에 다음 두 가지를 추가한다:

```
1. 차원별 보간 비율 γ(r_i)를 [0,1] 범위에서 연속적으로 설정
   - r_i = (L_train / (2π/θ_i')): 파장 대 학습 길이 비율
   - γ = 0이면 순수 보간, γ = 1이면 순수 외삽

2. Attention logit에 온도 보정 t를 적용
   - √(1 + 0.1 · ln(s))  (s = L_target / L_train)
```

YaRN은 소량의 fine-tuning(~400 step)만으로 128K+ 문맥 창을 달성하며, LLaMA-2 및 Mistral에서 검증되었다.

### 3.7 Flash Attention

#### 3.7.1 문제: IO-Bound Attention

표준 attention 구현은 다음 단계를 순차 실행한다:

```
1. S = QK^T ∈ ℝ^{n×n}          # HBM에 저장 (O(n²))
2. P = softmax(S)               # HBM에서 읽고 HBM에 저장
3. O = PV                       # HBM에서 읽어 계산
```

A100 GPU 기준, HBM 대역폭은 ~2TB/s이고 SRAM 대역폭은 ~19TB/s이다. 위 과정에서 n×n 크기의 중간 행렬 S, P를 HBM에 쓰고 읽는 IO가 병목이며, 실제 FLOP 활용률은 이론 최대치의 일부에 불과하다.

#### 3.7.2 Flash Attention v1의 해결

Dao et al. (2022)는 tiling과 online softmax 재계산을 결합한 IO-aware 알고리즘을 제안하였다:

```
IO 복잡도 분석:
  표준 attention: Θ(nd + n²)  HBM 접근
  Flash Attention: Θ(n²d² / M)  HBM 접근

  여기서 M = SRAM 크기 (A100: ~20MB)
  n²d² / M << n² + nd  (M이 충분히 크면)
```

알고리즘의 핵심 단계:

```
Q, K, V를 블록 B_r × B_c로 분할 (B_r, B_c ~ √M)

for each K_j, V_j block:
  for each Q_i block:
    1. SRAM에 Q_i, K_j, V_j 로드
    2. S_ij = Q_i K_j^T / √d_k  (SRAM에서 계산)
    3. Online softmax로 부분 결과 누적:
       m_new = max(m_old, rowmax(S_ij))
       l_new = e^{m_old - m_new} · l_old + rowsum(e^{S_ij - m_new})
       O_i = diag(l_old/l_new)^{-1} · e^{m_old-m_new} · O_i_old
            + diag(l_new)^{-1} · e^{S_ij - m_new} · V_j
    4. 중간 n×n 행렬을 HBM에 저장하지 않음
```

**효과:** 메모리 O(n²) → O(n), 벽시계 시간 2-4배 향상, 수학적으로 표준 attention과 정확히 동일한 출력을 보장한다.

#### 3.7.3 Flash Attention v2

Dao (2023)는 v1의 비효율을 개선하여 다음을 달성하였다:

1. **외부 루프와 내부 루프 순서 교환:** Q 블록을 외부 루프, KV 블록을 내부 루프로 변경하여, 비공유 메모리에서의 공유 메모리로의 쓰기 횟수를 감소시켰다.
2. **Warp 수준 병렬화:** 이전 버전의 thread block 내 동기화 오버헤드를 warp 간 작업 분할로 제거하였다.
3. **Causal masking 최적화:** 인과적 마스크가 적용되는 블록만 선택적으로 계산하여, 약 50%의 불필요한 연산을 제거하였다.

이론적 최대 FLOP 대비 도달률: v1 ~30-50%, v2 ~50-73% (A100 기준).

#### 3.7.4 Flash Attention v3

Shah et al. (2024)는 NVIDIA Hopper 아키텍처(H100)의 하드웨어 특성을 활용한 v3를 제안하였다:

1. **비동기 실행:** WGMMA(Warp Group Matrix Multiply-Accumulate)와 TMA(Tensor Memory Accelerator)를 오버랩하여 producer-consumer 파이프라인을 구현하였다.
2. **FP8 지원:** 저정밀 연산을 incoherent processing(랜덤 직교 변환 적용 후 양자화)과 결합하여 정밀도 손실을 최소화하였다.
3. **Block quantization:** 블록 단위 동적 스케일링으로 FP8의 좁은 동적 범위를 보완하였다.

H100에서 v2 대비 1.5-2배 추가 속도 향상을 달성하며, 이론적 최대 FLOP의 75%에 도달하였다.

### 3.8 Ring Attention

**문제:** 단일 GPU의 메모리에 담을 수 없는 초장문 시퀀스(수백만 토큰)를 처리해야 한다. 데이터 병렬화는 시퀀스를 분할하지 않으므로, 시퀀스 차원의 분산이 필요하다.

**해결:** Liu et al. (2023)는 시퀀스를 여러 디바이스에 분산하고, KV 블록을 링(ring) 토폴로지로 순환시키는 Ring Attention을 제안하였다:

```
N개 디바이스, 시퀀스를 N개 청크로 분할

Step 1: 디바이스 i는 Q_i를 보유하고, 로컬 K_i, V_i로 부분 attention 계산
Step 2: K_i, V_i를 다음 디바이스로 전송 (ring 통신)
Step 3: 수신한 K_{i-1}, V_{i-1}로 추가 부분 attention 계산
...
Step N: 모든 KV 블록을 순회한 뒤 online softmax로 최종 결과 통합
```

핵심은 attention 계산과 통신을 오버랩할 수 있다는 점이다. attention의 내부 루프 계산 시간이 KV 블록 전송 시간보다 크면, 통신 비용이 완전히 은닉된다. 이론적으로 디바이스 수에 비례하여 문맥 길이를 무한히 확장할 수 있다.

**새로운 문제:** 인과적 마스킹이 적용되는 경우 일부 디바이스의 계산량이 불균형하며, 통신 대역폭이 부족하면 은닉이 불완전하여 성능이 저하된다.

### 3.9 Sparse Attention

**문제:** 표준 attention의 O(n²) 복잡도는 매우 긴 시퀀스에서 비실용적이다.

**해결:** 전체 attention 행렬 대신 일부 위치 쌍만 계산하는 sparse attention 패턴을 사용한다. 대표적 기법은 다음과 같다:

1. **Sparse Transformer (Child et al., 2019):** 고정된 희소 패턴(strided + local) 조합으로 O(n√n) 복잡도를 달성하였다.
2. **Longformer (Beltagy et al., 2020):** 슬라이딩 윈도우(local attention) + 소수의 글로벌 토큰([CLS] 등)을 결합하여 O(n) 복잡도를 달성하였다.
3. **BigBird (Zaheer et al., 2020):** 랜덤 + 윈도우 + 글로벌 attention의 조합이 full attention의 표현력을 이론적으로 근사할 수 있음을 증명하였다.

**새로운 문제:** 고정 패턴은 데이터 분포에 최적이 아닐 수 있으며, 커스텀 CUDA 커널 없이는 하드웨어 효율이 낮다. Flash Attention의 등장으로, 중간 길이(~128K) 시퀀스에서는 dense attention + Flash Attention이 sparse attention보다 더 실용적인 선택이 되었다.

### 3.10 Linear Attention

**문제:** O(n²) 복잡도를 O(n)으로 근본적으로 줄일 수 있는가?

**해결:** Katharopoulos et al. (2020)는 softmax를 커널 함수 φ(·)로 대체하여 연산 순서를 변경하는 linear attention을 제안하였다:

```
표준: O_i = Σ_j [softmax(q_i^T k_j)] v_j^T       → O(n²d)

Linear: softmax(q^T k) ≈ φ(q)^T φ(k)
  → O_i = φ(q_i)^T [Σ_j φ(k_j) v_j^T]
  → S = Σ_j φ(k_j) v_j^T 를 점진적으로 누적  → O(nd²)
```

n >> d인 경우 O(nd²) << O(n²d)이므로, 시퀀스 길이에 대해 선형 복잡도를 달성한다.

**새로운 문제:** softmax attention 대비 표현력이 부족하여, 특히 retrieval-heavy 태스크에서 성능 격차가 유의미하다. 이를 보완하기 위한 연구(RetNet, RWKV, Mamba 등)가 진행되고 있으나, 2024-2025년 기준 최고 성능 LLM은 여전히 softmax attention을 사용한다.

### 3.11 Transformer의 부수 구성 요소

#### 3.11.1 SwiGLU Activation

**문제:** 표준 Transformer의 FFN은 ReLU를 사용하나, 이후 GELU가 더 나은 성능을 보였고, 더 나은 활성화 함수의 여지가 존재한다.

**해결:** Shazeer (2020)는 Gated Linear Unit(GLU)의 변형으로 SwiGLU를 제안하였다:

```
FFN_SwiGLU(x) = (Swish_β(xW_1) ⊙ xW_3) W_2

여기서 Swish_β(x) = x · σ(βx), ⊙은 원소별 곱
일반적으로 β = 1 (= SiLU)
```

표준 FFN이 2개의 가중치 행렬을 사용하는 반면, SwiGLU는 3개(W_1, W_2, W_3)를 사용한다. 파라미터 수를 동일하게 맞추기 위해 hidden dimension을 2/3로 줄이는 것이 관례이다 (예: 4d → 8d/3). 실험적으로 동일 파라미터 수에서 ReLU, GELU 대비 일관된 perplexity 개선이 관측되었다.

LLaMA, Mistral, Qwen, Gemma 등 2023년 이후의 거의 모든 주요 LLM이 SwiGLU를 채택하고 있다.

#### 3.11.2 RMSNorm

**문제:** LayerNorm은 평균과 분산을 모두 계산하며, re-centering(평균 제거)과 re-scaling(분산 정규화) 두 연산을 수행한다.

**해결:** Zhang & Sennrich (2019)는 re-centering이 불필요하며 re-scaling만으로 충분하다는 가설 하에 RMSNorm을 제안하였다:

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

RMSNorm(x) = γ · x / √(Σ x_i² / d + ε)
           = γ · x / RMS(x)
```

RMSNorm은 평균 계산을 제거하여 약 7-10%의 연산량을 절약하면서, 학습 품질에는 영향이 없다. LLaMA를 포함한 대부분의 현대 LLM이 LayerNorm 대신 RMSNorm을 사용한다.

#### 3.11.3 Pre-Norm vs Post-Norm

**Post-Norm (원본 Transformer):**
```
x' = LayerNorm(x + SubLayer(x))
```

**Pre-Norm (GPT-2 이후 표준):**
```
x' = x + SubLayer(LayerNorm(x))
```

**문제:** Post-Norm은 잔차 연결 이후에 정규화하므로, 깊은 모델에서 학습 초기에 기울기 불안정이 발생하여 learning rate warmup이 필수적이다.

**해결:** Pre-Norm은 sublayer 입력을 먼저 정규화하여, 잔차 경로의 기울기 크기를 안정화한다. Xiong et al. (2020)은 Pre-Norm이 이론적으로 warmup 없이도 안정적인 학습이 가능함을 증명하였다.

**새로운 문제:** Pre-Norm은 깊은 층에서 잔차 기여도가 감소하는 "representation collapse" 경향이 있으며, 일부 연구는 Post-Norm이 최종 성능에서 우위를 가진다고 보고한다. DeepSeek-V2는 이를 절충한 DeepNorm 변형을 사용한다.

### 3.12 현대 LLM 아키텍처 레시피 (2024-2025)

2024-2025년 기준 최고 성능 LLM들이 수렴한 아키텍처 구성 요소를 종합하면 다음과 같다:

| 구성 요소 | 표준 선택 | 채택 모델 | 대안 |
|-----------|----------|----------|------|
| **Attention** | GQA + RoPE | LLaMA-3, Mistral, Qwen-2 | MLA (DeepSeek-V2/V3) |
| **FFN Activation** | SwiGLU | LLaMA-3, Mistral, Gemma | GeGLU (일부) |
| **Normalization** | RMSNorm (Pre-Norm) | LLaMA-3, Mistral | DeepNorm (DeepSeek) |
| **Vocabulary** | BPE, 32K-128K tokens | 대부분 | SentencePiece |
| **Embedding** | Untied (입출력 분리) | LLaMA-3 | Tied (Gemma) |
| **Position** | RoPE + YaRN/NTK 확장 | LLaMA-3, Qwen-2 | ALiBi (BLOOM, MPT) |
| **KV Cache** | GQA (8 KV heads) | LLaMA-3, Mistral | MLA (DeepSeek) |
| **Attention Kernel** | Flash Attention v2/v3 | 사실상 모든 모델 | - |
| **FFN 크기** | ~8/3 × d_model | SwiGLU 보정 | 4 × d_model (ReLU) |

이 레시피는 경험적 최적화의 결과이며, 각 구성 요소의 선택은 독립적으로 ablation된 연구 결과에 기반한다. 특히 GQA + RoPE + SwiGLU + RMSNorm(Pre-Norm)의 조합은 LLaMA (Touvron et al., 2023)에서 처음 통합되어, 이후 사실상의 표준(de facto standard)이 되었다.

---

## 4. 기법 진화의 인과적 흐름

Attention mechanism과 Transformer architecture의 진화는 단일 선형 경로가 아니라, 여러 독립적 문제-해결 축이 병렬로 진행된 구조이다. 각 축의 인과적 흐름을 정리하면 다음과 같다.

### 4.1 시퀀스 모델링 패러다임 전환

```
RNN/LSTM: 순차 처리, 장거리 의존성 한계
  → Bahdanau Attention (ICLR 2015): 동적 문맥 참조로 병목 해소
    → 그러나 여전히 RNN 기반 → 병렬화 불가
      → Self-Attention (NeurIPS 2017): 순환 구조 완전 제거
        → Transformer: 병렬화 가능, O(1) 경로 길이
          → 그러나 O(n²) 복잡도 → 긴 시퀀스 처리 한계
```

이 전환의 핵심은, attention이 "RNN의 보조 장치"에서 "유일한 시퀀스 모델링 메커니즘"으로 승격된 패러다임 전환이다.

### 4.2 KV Cache 효율화 축

```
MHA (NeurIPS 2017): 각 head가 독립 KV → 풍부한 표현력
  → 추론 시 KV cache가 메모리 지배적
    → MQA (Shazeer, 2019): 모든 head가 KV 공유 → 1/h 캐시
      → 표현력 감소 우려
        → GQA (EMNLP 2023): 그룹별 KV 공유 → MHA-MQA 절충
          → 여전히 seq_len에 선형
            → MLA (DeepSeek-V2, 2024): 저차원 잠재 공간 압축
              → 93.3% 캐시 감소, 복원 연산 추가 비용
```

### 4.3 위치 인코딩 축

```
Sinusoidal (2017): 고정 삼각함수 → 이론적 외삽 가능
  → Learned (GPT-2, 2019): 더 유연하지만 외삽 불가
    → RoPE (2021): 회전 행렬로 상대 위치 → 외삽 가능
      → 학습 길이 초과 시 성능 급락
        → Position Interpolation (2023): 선형 축소 → 해상도 손실
          → NTK-aware (2023): 주파수별 적응적 처리
            → YaRN (ICLR 2024): NTK + 온도 보정 → 128K+ 달성
ALiBi (2022): 거리 편향 방식 → 우수한 외삽
  → 장거리 정보 과도한 억제 → RoPE 계열이 주류로 수렴
```

### 4.4 연산/메모리 효율화 축

```
표준 attention: O(n²) 메모리, HBM IO 병목
  → Flash Attention v1 (NeurIPS 2022): tiling + online softmax → O(n) 메모리
    → GPU 활용률 30-50%에 불과
      → Flash Attention v2 (ICLR 2024): 루프 재구성 + warp 병렬화 → 50-73%
        → Hopper GPU 특화 최적화 필요
          → Flash Attention v3 (2024): WGMMA + TMA + FP8 → 75%

Sparse Attention (2019): 희소 패턴으로 O(n√n) 또는 O(n)
  → 하드웨어 비효율, Flash Attention에 밀림 (중간 길이 시퀀스)
    → 초장문에서만 유효

Ring Attention (ICLR 2024): 다중 디바이스 시퀀스 분산
  → 통신-연산 오버랩으로 이론적 무한 확장
    → 인과적 마스킹 시 부하 불균형

Linear Attention (2020): softmax → 커널 근사 → O(nd²)
  → 표현력 부족 → Mamba/RWKV 등 SSM/RNN 하이브리드로 진화
    → 2025년 기준 여전히 softmax attention이 최고 성능
```

### 4.5 아키텍처 구성 요소 수렴

```
원본 Transformer (2017): Post-Norm + ReLU FFN + Sinusoidal + MHA
  → GPT-2 (2019): Pre-Norm + GELU + Learned PE
    → GPT-3 (2020): 스케일 업, 아키텍처 동일
      → PaLM (2022): SwiGLU 도입, RoPE 미사용
        → LLaMA (2023): Pre-Norm RMSNorm + SwiGLU + RoPE + GQA
          → 사실상의 표준 레시피 확립
            → 이후 Mistral, Qwen, Yi, Gemma 등이 동일 구성 채택
```

이 수렴 과정의 특징은, 각 구성 요소가 독립적으로 ablation 연구되었고, 최종 조합이 경험적 최적에 해당한다는 것이다. LLaMA의 기여는 개별 기법의 발명이 아니라, 이들을 최적 조합으로 통합하고 오픈소스로 공개하여 사실상의 표준을 확립한 데 있다.

### 4.6 미해결 문제와 향후 방향

1. **O(n²) 복잡도의 근본적 해결:** Linear attention, SSM(Mamba), RWKV 등이 시도되고 있으나, softmax attention의 표현력을 완전히 대체하지 못하고 있다. 하이브리드 접근(Jamba: Mamba + Attention)이 유망한 중간 지점이다.

2. **무한 문맥 창:** Ring Attention과 YaRN은 기술적 확장을 가능케 하지만, 모델이 실제로 긴 문맥을 효과적으로 활용하는지("lost in the middle" 문제)는 별개의 연구 주제이다.

3. **추론 효율:** MLA와 GQA는 KV cache를 줄이지만, 토큰 생성의 근본적 순차성(autoregressive bottleneck)은 해결하지 못한다. Speculative decoding, Medusa 등의 병렬 생성 기법이 이를 보완한다.

4. **하드웨어 공동 설계:** Flash Attention v3가 Hopper 아키텍처에 특화된 것처럼, 알고리즘과 하드웨어의 공동 최적화가 점점 중요해지고 있다. 차세대 GPU(Blackwell)와 TPU에 대한 적응이 지속적으로 필요하다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 1 | **Neural Machine Translation by Jointly Learning to Align and Translate** | Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio | **ICLR 2015** | https://arxiv.org/abs/1409.0473 |
| 2 | **Attention Is All You Need** | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin | **NeurIPS 2017** | https://arxiv.org/abs/1706.03762 |
| 3 | **Generating Long Sequences with Sparse Transformers** | Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever | arXiv 2019 | https://arxiv.org/abs/1904.10509 |
| 4 | **Fast Transformer Decoding: One Write-Head is All You Need** | Noam Shazeer | arXiv 2019 | https://arxiv.org/abs/1911.02150 |
| 5 | **Language Models are Unsupervised Multitask Learners (GPT-2)** | Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever | OpenAI Technical Report 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| 6 | **GLU Variants Improve Transformer** | Noam Shazeer | arXiv 2020 | https://arxiv.org/abs/2002.05202 |
| 7 | **Longformer: The Long-Document Transformer** | Iz Beltagy, Matthew E. Peters, Arman Cohan | arXiv 2020 | https://arxiv.org/abs/2004.05150 |
| 8 | **Big Bird: Transformers for Longer Sequences** | Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed | **NeurIPS 2020** | https://arxiv.org/abs/2007.14062 |
| 9 | **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** | Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pitas, François Fleuret | **ICML 2020** | https://arxiv.org/abs/2006.16236 |
| 10 | **Language Models are Few-Shot Learners (GPT-3)** | Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Siber, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei | **NeurIPS 2020** | https://arxiv.org/abs/2005.14165 |
| 11 | **On Layer Normalization in the Transformer Architecture** | Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tieyan Liu | **ICML 2020** | https://arxiv.org/abs/2002.04745 |
| 12 | **Root Mean Square Layer Normalization** | Biao Zhang, Rico Sennrich | **NeurIPS 2019** | https://arxiv.org/abs/1910.07467 |
| 13 | **RoFormer: Enhanced Transformer with Rotary Position Embedding** | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu | **Neurocomputing 2024** | https://arxiv.org/abs/2104.09864 |
| 14 | **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)** | Ofir Press, Noah A. Smith, Mike Lewis | **ICLR 2022** | https://arxiv.org/abs/2108.12409 |
| 15 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré | **NeurIPS 2022** | https://arxiv.org/abs/2205.14135 |
| 16 | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | Tri Dao | **ICLR 2024** | https://arxiv.org/abs/2307.08691 |
| 17 | **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** | Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao | arXiv 2024 | https://arxiv.org/abs/2407.08691 |
| 18 | **PaLM: Scaling Language Modeling with Pathways** | Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann et al. | **JMLR 2023** | https://arxiv.org/abs/2204.02311 |
| 19 | **LLaMA: Open and Efficient Foundation Language Models** | Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample | arXiv 2023 | https://arxiv.org/abs/2302.13971 |
| 20 | **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** | Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale et al. | arXiv 2023 | https://arxiv.org/abs/2307.09288 |
| 21 | **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** | Joshua Ainslie, James Lee-Thorp, Michal de Jong, Yinfei Yang, Cuthan Saharia, David Grangier | **EMNLP 2023** | https://arxiv.org/abs/2305.13245 |
| 22 | **Extending Context Window of Large Language Models via Positional Interpolation** | Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian | arXiv 2023 | https://arxiv.org/abs/2306.15595 |
| 23 | **NTK-Aware Scaled RoPE Allows LLaMA Models to Have Extended Context** | bloc97 | Reddit/GitHub 2023 | https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ |
| 24 | **YaRN: Efficient Context Window Extension of Large Language Models** | Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole | **ICLR 2024** | https://arxiv.org/abs/2309.00071 |
| 25 | **Ring Attention with Blockwise Transformers for Near-Infinite Context** | Hao Liu, Matei Zaharia, Pieter Abbeel | **ICLR 2024** | https://arxiv.org/abs/2310.01889 |
| 26 | **Mistral 7B** | Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed | arXiv 2023 | https://arxiv.org/abs/2310.06825 |
| 27 | **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** | DeepSeek-AI | arXiv 2024 | https://arxiv.org/abs/2405.04434 |
| 28 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | Albert Gu, Tri Dao | arXiv 2023 | https://arxiv.org/abs/2312.00752 |
| 29 | **RWKV: Reinventing RNNs for the Transformer Era** | Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski et al. | **EMNLP 2023 Findings** | https://arxiv.org/abs/2305.13048 |
| 30 | **Jamba: A Hybrid Transformer-Mamba Language Model** | Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, Yoav Shoham | arXiv 2024 | https://arxiv.org/abs/2403.19887 |
| 31 | **Sequence to Sequence Learning with Neural Networks** | Ilya Sutskever, Oriol Vinyals, Quoc V. Le | **NeurIPS 2014** | https://arxiv.org/abs/1409.3215 |
| 32 | **Effective Approaches to Attention-based Neural Machine Translation** | Minh-Thang Luong, Hieu Pham, Christopher D. Manning | **EMNLP 2015** | https://arxiv.org/abs/1508.04025 |
| 33 | **Qwen2 Technical Report** | Qwen Team (Alibaba) | arXiv 2024 | https://arxiv.org/abs/2407.10671 |
| 34 | **The Llama 3 Herd of Models** | Meta AI | arXiv 2024 | https://arxiv.org/abs/2407.21783 |
| 35 | **Gemma: Open Models Based on Gemini Research and Technology** | Gemma Team, Google DeepMind | arXiv 2024 | https://arxiv.org/abs/2403.08295 |
| 36 | **Lost in the Middle: How Language Models Use Long Contexts** | Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang | **TACL 2024** | https://arxiv.org/abs/2307.03172 |
| 37 | **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao | **ICML 2024** | https://arxiv.org/abs/2401.10774 |
| 38 | **DeepSeek-V3 Technical Report** | DeepSeek-AI | arXiv 2024 | https://arxiv.org/abs/2412.19437 |
