# Mixture of Experts (MoE, 전문가 혼합)

## 1. 기법의 정의

Mixture of Experts(MoE)는 단일 모델 내에 다수의 하위 네트워크(전문가, expert)를 배치하고, 입력에 따라 일부 전문가만 선택적으로 활성화하는 조건부 연산(conditional computation) 기법이다. 핵심 구성 요소는 (1) N개의 전문가 네트워크 $\{E_1, E_2, \ldots, E_N\}$와 (2) 입력 토큰 $x$에 대해 어떤 전문가를 활성화할지 결정하는 게이팅 네트워크(gating network) $G(x)$이다.

MoE 레이어의 출력은 다음과 같이 정의된다:

$$
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

여기서 $G(x) \in \mathbb{R}^N$은 게이팅 함수의 출력 벡터이며, sparse gating의 경우 상위 K개를 제외한 나머지 요소가 0으로 설정된다. 이를 통해 총 파라미터 수는 $O(N \cdot d_{expert})$로 증가하지만, 입력당 활성 연산량은 $O(K \cdot d_{expert})$에 머물러 파라미터 효율성과 연산 효율성을 동시에 달성할 수 있다.

현대 Transformer 기반 MoE에서는 각 Transformer 블록의 Feed-Forward Network(FFN)를 MoE 레이어로 교체하는 것이 표준적 설계이다. Self-Attention 레이어는 모든 토큰이 공유하며, FFN만 전문가로 분화시킨다.

---

## 2. 기존 기법의 한계와 MoE 등장 배경

### 2.1 Dense 모델의 연산량 문제

Kaplan et al. (2020)이 제시한 신경망 스케일링 법칙(Neural Scaling Laws)에 따르면, 언어 모델의 성능은 모델 파라미터 수 $N$, 데이터 크기 $D$, 연산량 $C$의 멱법칙(power law) 관계를 따른다. 이는 성능 향상을 위해 모델 크기를 지속적으로 확장해야 함을 의미한다.

**문제:** Dense 모델에서는 연산량이 파라미터 수에 정비례한다. 즉, $N$개의 파라미터를 가진 Dense 모델의 순방향 연산량은 대략 $2N$ FLOPs이다. GPT-3 175B의 경우 단일 토큰 추론에 약 350 TFLOPs가 소요되며, 이는 하드웨어 비용과 추론 지연(latency) 측면에서 실용적 한계를 초래한다.

**해결 방향:** 파라미터 수(모델 용량)와 연산량(추론 비용)을 분리(decouple)할 수 있다면, 대규모 파라미터의 표현력은 유지하면서 연산량은 소규모 모델 수준으로 억제할 수 있다. 이것이 MoE의 핵심 동기이다.

```
Dense 70B 모델:   70B 파라미터 × 100% 활성화 = ~140 TFLOPs/token
MoE 46.7B 모델:   46.7B 파라미터 × ~28% 활성화 = ~25.8 TFLOPs/token (Mixtral 8x7B)
MoE 671B 모델:    671B 파라미터 × ~5.5% 활성화 = ~74 TFLOPs/token (DeepSeek-V3)
→ Dense 대비 동일 연산량에서 수 배 이상의 파라미터 용량을 확보
```

### 2.2 조건부 연산의 이론적 근거

Bengio et al. (2013)은 조건부 연산의 이론적 동기를 체계화하였다. 모든 입력에 대해 동일한 연산 경로를 사용하는 Dense 모델은, 쉬운 입력에 대해 불필요한 연산을 수행하고, 어려운 입력에 대해서는 연산이 부족할 수 있다. MoE는 입력의 특성에 따라 활성화되는 전문가를 달리함으로써 적응적 연산 할당(adaptive computation)을 실현한다.

**새로운 문제:** 조건부 연산은 이산적(discrete) 라우팅 결정을 포함하므로, 표준 역전파(backpropagation)로 학습하기 어렵다. 또한 다수의 전문가 중 일부만 활성화되므로, 활성화되지 않는 전문가의 파라미터는 갱신되지 않아 전문가 붕괴(expert collapse) 문제가 발생한다.

---

## 3. 주요 기법 상세

### 3.1 Original MoE (1991)

Jacobs et al. (1991)은 "Adaptive Mixtures of Local Experts"에서 MoE 프레임워크를 최초로 제안하였다. 게이팅 네트워크는 softmax 기반 선형 분류기로 구성되며, 각 전문가는 입력 공간의 특정 부분 영역(local region)을 담당하도록 경쟁적 학습(competitive learning)을 수행한다.

**게이팅 함수:**

$$
G(x)_i = \frac{\exp(W_g^{(i)} \cdot x)}{\sum_{j=1}^{N} \exp(W_g^{(j)} \cdot x)}
$$

**문제:** Dense softmax gating은 모든 전문가에 0이 아닌 가중치를 부여하므로, N이 커지면 연산량이 선형적으로 증가한다. 또한 당시 하드웨어 규모에서는 소수의 전문가(2~8개)만 실험할 수 있었으며, 대규모 확장은 불가능하였다.

### 3.2 Sparsely-Gated MoE Layer (2017)

**문제:** Dense gating으로는 수천 개의 전문가를 사용할 수 없다.

**해결:** Shazeer et al. (2017)은 Top-K sparse gating을 도입하여 이 문제를 해결하였다. 게이팅 함수의 출력에서 상위 K개만 남기고 나머지를 0으로 설정한다.

$$
G(x) = \text{Softmax}(\text{TopK}(H(x), K))
$$

$$
H(x)_i = W_g \cdot x + \epsilon \cdot \text{Softplus}(W_{noise} \cdot x)_i, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

여기서 $\text{TopK}(\cdot, K)$는 상위 K개 값을 유지하고 나머지를 $-\infty$로 설정하는 연산이다. 노이즈 항 $\epsilon$은 학습 초기에 탐색(exploration)을 촉진하여 전문가 붕괴를 완화한다.

이 연구에서는 LSTM 기반 언어 모델에 MoE 레이어를 삽입하여, 4096개의 전문가와 Top-4 gating으로 1000억 파라미터급 모델을 구현하였다. 당시 기준 연산량 대비 최고 성능을 달성하였다.

**부하 균형 손실(Load Balancing Loss):**

전문가 간 부하 불균형을 방지하기 위해, 다음과 같은 보조 손실(auxiliary loss)을 도입하였다:

$$
\mathcal{L}_{balance} = w \cdot CV(L)^2
$$

여기서 $CV(L)$은 전문가별 부하 $L_i$의 변동 계수(coefficient of variation)이다. 이 손실을 주 학습 손실에 가산하여 전문가 활용의 균일성을 유도한다.

**새로운 문제:** (1) 부하 균형 손실의 가중치 $w$ 튜닝이 민감하다. (2) 분산 학습 환경에서 전문가를 여러 디바이스에 분산 배치할 때 통신 오버헤드가 발생한다. (3) Top-K 연산 자체가 미분 불가능하여, straight-through estimator 등의 근사가 필요하다.

### 3.3 GShard (2021)

**문제:** MoE를 Transformer에 적용하고 수천 디바이스로 분산 학습하는 방법론이 부재하였다.

**해결:** Lepikhin et al. (2021)은 GShard에서 600B 파라미터 MoE Transformer를 2048개 TPU에서 학습하는 시스템을 제시하였다. 핵심 기여는 다음과 같다:

1. **Expert Parallelism:** 전문가를 서로 다른 디바이스에 배치하고, all-to-all 통신으로 토큰을 해당 전문가에 라우팅한다.
2. **Capacity Factor (CF):** 전문가당 처리 가능한 최대 토큰 수를 제한하여 메모리 사용량을 예측 가능하게 만든다.

$$
C_i = CF \cdot \frac{T}{N}
$$

여기서 $T$는 배치 내 총 토큰 수, $N$은 전문가 수이다. $CF > 1$은 약간의 여유를 허용하며, 용량을 초과하는 토큰은 드롭된다(overflow dropping).

3. **Random Routing (보조 전문가):** Top-2 gating에서 1순위 전문가는 결정적으로, 2순위 전문가는 게이팅 가중치에 비례하는 확률로 라우팅한다.

**새로운 문제:** 용량 초과 시 토큰이 드롭되면 정보 손실이 발생한다. 또한 CF 값의 설정이 효율성과 성능 간 트레이드오프를 결정하며, 최적값을 찾기 어렵다.

### 3.4 Switch Transformer (2022)

**문제:** Top-2 gating은 여전히 연산량이 크고, 라우팅 복잡도를 줄여 학습 처리량(throughput)을 극대화할 방법이 필요하였다.

**해결:** Fedus et al. (2022)은 극단적 단순화로 Top-1 gating을 채택하였다. 각 토큰은 정확히 하나의 전문가에만 라우팅된다.

$$
G(x) = \text{Softmax}(W_g \cdot x), \quad i^* = \arg\max_i G(x)_i, \quad y = G(x)_{i^*} \cdot E_{i^*}(x)
$$

**부하 균형 손실 재정의:**

$$
\mathcal{L}_{balance} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

여기서 $f_i$는 전문가 $i$에 라우팅된 토큰의 비율, $P_i$는 전문가 $i$에 대한 게이팅 확률의 평균, $\alpha$는 가중치 하이퍼파라미터이다. 이 공식은 $f_i$와 $P_i$가 모두 균일할 때 최소값 $\alpha$를 가진다.

Switch Transformer는 T5-Base 대비 동일 연산량에서 7배 빠른 사전 학습 속도를 달성하였으며, T5-XXL(11B)과 동등한 성능을 1/10 연산량으로 도달하였다. 2048개 전문가를 사용한 1.6T 파라미터 모델까지 확장하였다.

**새로운 문제:** (1) Top-1 라우팅은 단일 전문가의 표현력에 의존하므로, 복합적 입력에 대한 정보 손실 가능성이 존재한다. (2) bfloat16 학습 시 라우터의 수치 불안정성이 관찰되었으며, 라우터만 float32로 연산하는 selective precision이 필요하였다. (3) 파인튜닝 시 Dense 모델 대비 과적합이 심하다.

### 3.5 ST-MoE: 안정적이고 전이 가능한 MoE 설계 (2022)

**문제:** MoE 모델의 학습 불안정성(training instability)이 체계적으로 분석되지 않았다.

**해결:** Zoph et al. (2022)은 ST-MoE에서 MoE 학습 불안정성의 원인을 실험적으로 규명하고, 안정화 기법을 제시하였다.

1. **Router z-loss:** 라우터 로짓의 크기를 정규화하여 수치 불안정성을 방지한다.

$$
\mathcal{L}_{z} = \frac{1}{B} \sum_{x \in B} \left( \log \sum_{i=1}^{N} \exp(z_i(x)) \right)^2
$$

여기서 $z_i(x)$는 softmax 적용 전의 라우터 로짓이다. 이 손실은 로짓 값이 과도하게 커지는 것을 억제하여, softmax 연산의 수치 안정성을 보장한다.

2. **Encoder-Decoder 구조에서 MoE 레이어 배치:** 인코더의 마지막 레이어와 디코더의 모든 레이어에 MoE를 배치하는 것이 최적임을 실험적으로 확인하였다.

3. **파인튜닝 안정화:** 드롭아웃 비율 증가, 학습률 감소 등의 정규화 전략이 MoE 파인튜닝에 효과적임을 보였다.

**새로운 문제:** ST-MoE의 분석은 인코더-디코더 구조에 한정되어 있으며, 디코더 전용(decoder-only) 구조에서의 일반화는 검증되지 않았다.

### 3.6 Hash Layer (2021)

**문제:** 학습 가능한 라우터는 부하 균형 문제를 유발하고 추가 파라미터를 필요로 한다.

**해결:** Roller et al. (2021)은 Hash Layer에서 고정 해시 함수로 토큰을 전문가에 매핑하는 방식을 제안하였다. 토큰 ID를 기반으로 결정론적 해싱을 수행하므로, 라우터 파라미터가 불필요하고 부하 균형이 자동으로 보장된다.

$$
\text{expert\_id}(x) = \text{hash}(\text{token\_id}(x)) \mod N
$$

실험 결과, 해시 기반 라우팅이 학습된 라우터와 비교 가능한 성능을 보이는 경우가 존재하였으나, 문맥 의존적 라우팅이 불가능하다는 근본적 한계가 있다.

**새로운 문제:** 동일 토큰이 문맥에 관계없이 항상 같은 전문가에 라우팅되므로, 다의어(polysemy) 등 문맥 의존적 현상을 처리하지 못한다.

### 3.7 Expert Choice Routing (2022)

**문제:** Token-choice 라우팅(토큰이 전문가를 선택)은 부하 불균형과 토큰 드롭 문제를 유발한다.

**해결:** Zhou et al. (2022)은 Expert Choice Routing에서 라우팅의 방향을 역전시켜, 전문가가 자신이 처리할 토큰을 선택하도록 하였다.

$$
S = \text{Softmax}(W_g \cdot X^T) \in \mathbb{R}^{N \times T}
$$

각 전문가 $i$는 점수 행렬 $S$의 $i$번째 행에서 상위 $C$개 토큰을 선택한다. 여기서 $C = CF \cdot T / N$은 전문가당 용량이다.

**장점:** (1) 부하 균형이 구조적으로 보장된다(각 전문가가 정확히 $C$개 토큰을 처리). (2) 보조 손실이 불필요하다. (3) 중요한 토큰은 여러 전문가에 의해 선택될 수 있어 가변 연산량 할당이 가능하다.

**새로운 문제:** (1) 일부 토큰이 어떤 전문가에도 선택되지 않을 수 있다(token dropping). (2) 자기회귀(autoregressive) 디코딩에서는 미래 토큰과의 경쟁이 불가능하므로 적용이 제한적이다.

### 3.8 Mixtral 8x7B 및 8x22B (2024)

**문제:** 오픈소스 생태계에서 실용적인 MoE 모델이 부재하였다.

**해결:** Jiang et al. (2024)은 Mixtral 8x7B에서 8개의 전문가와 Top-2 gating을 적용한 46.7B 파라미터 모델을 공개하였다. 각 토큰당 12.9B 파라미터만 활성화되지만, LLaMA-2 70B와 동등하거나 우수한 성능을 달성하였다.

| 벤치마크 | Mixtral 8x7B | LLaMA-2 70B | GPT-3.5 |
|----------|-------------|-------------|---------|
| MMLU | 70.6 | 69.8 | 70.0 |
| HellaSwag | 86.7 | 87.3 | 85.5 |
| ARC-C | 66.0 | 64.6 | - |
| 활성 파라미터 | 12.9B | 70B | - |
| 추론 FLOPs 비율 | ~18% | 100% | - |

Mixtral 8x22B는 8개 전문가 × 22B로 141B 총 파라미터, 39B 활성 파라미터 구성이며 LLaMA-2 70B를 상회하는 성능을 보인다.

**구조적 특징:** Mistral 7B와 동일한 아키텍처에서 FFN만 MoE로 교체하였으며, Sliding Window Attention(SWA)을 결합하였다. 각 전문가의 FFN 차원은 Mistral 7B와 동일하다.

**새로운 문제:** 8개의 비교적 큰 전문가를 사용하므로, 전문가 조합의 다양성이 $\binom{8}{2} = 28$에 불과하다. 더 세밀한 전문가 분화가 가능한지에 대한 의문이 제기되었다.

### 3.9 DBRX (2024)

**문제:** Mixtral의 8개 전문가 구성은 라우팅 조합의 다양성이 제한적이다.

**해결:** Databricks의 DBRX는 16개 전문가에서 Top-4를 선택하는 fine-grained 구성을 채택하였다. 총 132B 파라미터 중 36B가 활성화된다. 전문가 수를 늘리고 활성 전문가 비율도 높여 $\binom{16}{4} = 1820$가지의 라우팅 조합을 확보하였다.

### 3.10 Qwen2-MoE (2024)

**문제:** 전문가 수 확장과 Shared Expert의 효과를 대규모로 검증한 사례가 부족하였다.

**해결:** Qwen 팀의 Qwen2-MoE(57B)는 64개 전문가에서 Top-8을 선택하며, 별도의 Shared Expert를 도입하였다. 14B 활성 파라미터로 LLaMA-3 70B에 근접하는 성능을 달성하였다. Shared Expert가 범용적 언어 지식을 담당하고, 라우팅 전문가가 도메인 특화 지식을 처리하는 구조이다.

### 3.11 DeepSeekMoE (2024)

**문제:** 기존 MoE 연구에서 전문가의 세분화(granularity) 수준과 전문화(specialization) 정도에 대한 체계적 분석이 부족하였다.

**해결:** Dai et al. (2024)은 DeepSeekMoE에서 두 가지 핵심 전략을 제안하였다:

1. **Fine-Grained Expert Segmentation:** 기존 $N$개의 큰 전문가 대신, 각 전문가를 $m$개로 분할하여 $mN$개의 작은 전문가를 구성하고 Top-$mK$를 선택한다. 총 활성 파라미터 수는 동일하지만, 라우팅 조합의 수가 기하급수적으로 증가한다.

$$
\text{기존: } \binom{N}{K} \text{ 조합} \quad \rightarrow \quad \text{Fine-grained: } \binom{mN}{mK} \text{ 조합}
$$

예를 들어, $N=8, K=2$일 때 $\binom{8}{2} = 28$이지만, $m=4$로 세분화하면 $\binom{32}{8} = 10,518,300$으로 조합의 다양성이 극적으로 증가한다.

2. **Shared Expert Isolation:** $N_s$개의 전문가를 shared expert로 지정하여 모든 토큰에 항상 활성화한다. 이는 범용적 지식(공통 문법, 일반 지식)을 shared expert에 집중시키고, routed expert는 전문 지식에 특화하도록 유도한다.

$$
y = \sum_{i=1}^{N_s} E_i^{(s)}(x) + \sum_{j=1}^{N_r} G(x)_j \cdot E_j^{(r)}(x)
$$

DeepSeekMoE 16B는 2B 활성 파라미터로 LLaMA-2 7B에 근접하는 성능을 보였다.

**새로운 문제:** Fine-grained expert가 매우 소규모이므로, 개별 전문가의 표현력이 제한적이다. 또한 전문가 수 증가에 따라 all-to-all 통신 오버헤드가 증가한다.

### 3.12 DeepSeek-V2 (2024)

**문제:** 전문가 세분화를 대규모 모델에 적용하고, 주의 메커니즘의 KV 캐시 비용을 동시에 절감해야 하였다.

**해결:** DeepSeek AI (2024a)는 DeepSeek-V2에서 MoE와 Multi-head Latent Attention(MLA)을 결합한 236B 파라미터 모델을 제시하였다. MoE 구성은 다음과 같다:

- 총 160개 routed expert + 2개 shared expert
- Top-6 선택, 21B 활성 파라미터
- Device-Limited Routing: 통신 오버헤드를 줄이기 위해 토큰이 라우팅되는 디바이스 수를 제한

DeepSeek-V2는 Mixtral 8x22B와 비교하여 동등한 성능을 42.5%의 연산량으로 달성하였다. MLA는 KV 캐시를 93.3% 절감하여, MoE의 연산 효율성과 MLA의 메모리 효율성이 시너지를 형성하였다.

**새로운 문제:** 160개 전문가의 분산 배치에서 all-to-all 통신이 여전히 병목이며, 학습 안정성 확보를 위한 추가 기법이 필요하였다.

### 3.13 DeepSeek-V3 (2024)

**문제:** MoE 모델의 학습 안정성, 부하 균형, 통신 효율을 동시에 개선해야 하였다.

**해결:** DeepSeek AI (2024b)는 DeepSeek-V3에서 671B 파라미터(37B 활성) 모델을 구현하며, 다음 혁신을 도입하였다:

1. **Auxiliary-Loss-Free Load Balancing:** 기존 보조 손실 기반 부하 균형은 주 학습 목표와 충돌할 수 있다. DeepSeek-V3는 각 전문가에 학습 가능한 바이어스 항 $b_i$를 부여하여, 라우팅 시에만 사용하고 게이팅 가중치 계산에는 포함시키지 않는다.

$$
i^* = \arg\max_i (G(x)_i + b_i), \quad \text{단, 출력 가중치는 } G(x)_{i^*} \text{만 사용}
$$

바이어스 $b_i$는 전문가 부하에 따라 동적으로 조정된다: 과부하 전문가의 바이어스를 감소시키고, 저부하 전문가의 바이어스를 증가시킨다.

2. **Multi-Token Prediction (MTP):** 각 위치에서 다음 1개가 아닌 2개 토큰을 예측하도록 학습하여, 데이터 효율성을 높인다.

3. **FP8 Mixed Precision Training:** 연산의 상당 부분을 FP8로 수행하여 학습 효율을 극대화하였다.

DeepSeek-V3는 14.8T 토큰으로 학습되었으며, 2048개 H800 GPU에서 약 2.788M GPU-hours를 소비하였다. 이는 GPT-4o 수준의 성능을 1/10 이하의 학습 비용으로 달성한 것이다.

| 벤치마크 | DeepSeek-V3 | GPT-4o | LLaMA-3.1 405B |
|----------|-------------|--------|----------------|
| MMLU | 88.5 | 88.7 | 88.6 |
| MATH 500 | 90.2 | 76.6 | 73.8 |
| HumanEval | 82.6 | 90.2 | 89.0 |
| 활성 파라미터 | 37B | ~200B(추정) | 405B |
| 학습 비용 | ~$5.6M | ~$100M+ | ~$30M+ |

### 3.14 MoE의 스케일링 법칙

**문제:** Dense 모델의 스케일링 법칙(Chinchilla 법칙 등)이 MoE에 직접 적용되는지 불명확하였다.

**해결:** Clark et al. (2022)은 "Unified Scaling Laws for Routed Language Models"에서 MoE 모델의 스케일링 법칙을 실험적으로 도출하였다. 핵심 발견은 다음과 같다:

1. MoE 모델의 성능은 전문가 수 $E$에 대해 로그 관계를 따른다:

$$
L(E) = L_{\text{dense}} - k \cdot \log(E)
$$

즉, 전문가 수를 2배로 늘려도 성능 향상은 로그적으로 감소한다. 이는 전문가 수 확장의 수확 체감(diminishing returns)을 의미한다.

2. 전문가 수 $E$개의 MoE 모델은 약 $E^{0.6}$배의 Dense 모델에 해당하는 "유효 파라미터 수"를 가진다. 즉, 64개 전문가 MoE는 약 $64^{0.6} \approx 12$배의 Dense 모델과 동등한 성능을 보인다.

Krajewski et al. (2024)는 이를 확장하여, MoE의 최적 학습 토큰 수 대비 전문가 수 비율에 대한 그래뉼러 스케일링 법칙을 제시하였다.

**새로운 문제:** 스케일링 법칙의 로그적 특성은 전문가 수를 무한히 늘리는 것이 비효율적임을 의미하며, 전문가의 "질"을 높이는 방향(fine-grained segmentation, shared expert 등)이 더 중요함을 시사한다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 라우팅 메커니즘의 진화

```
Dense Softmax Gating (1991, Jacobs et al.)
  │ 문제: 모든 전문가 활성화 → 연산 비효율
  ▼
Sparse Top-K Gating + Noise (2017, Shazeer et al.)
  │ 문제: 부하 불균형, 학습 불안정
  ▼
Top-2 + Capacity Factor + Random Routing (2021, GShard)
  │ 문제: 토큰 드롭, CF 튜닝 어려움
  ▼
Top-1 + 단순화된 부하 균형 손실 (2022, Switch Transformer)
  │ 문제: 단일 전문가 의존, 수치 불안정
  ▼
Expert Choice (역방향 라우팅) (2022, Zhou et al.)
  │ 문제: 자기회귀 디코딩 적용 제한
  ▼
Hash-based Fixed Routing (2021, Roller et al.)
  │ 문제: 문맥 의존적 라우팅 불가
  ▼
Auxiliary-Loss-Free Bias-based Routing (2024, DeepSeek-V3)
  │ 해결: 보조 손실 없이 바이어스로 부하 균형
  └ 현재 최선의 방법론
```

### 4.2 전문가 구조의 진화

```
소수의 큰 전문가 (8개, Mixtral)
  │ 문제: 라우팅 조합 제한 (28가지)
  ▼
중간 규모 전문가 (16개, DBRX)
  │ 개선: 1820가지 조합
  ▼
Fine-grained Expert Segmentation (160개, DeepSeekMoE)
  │ 개선: 수백만 가지 조합, 높은 전문화
  │ 문제: 개별 전문가 표현력 저하
  ▼
Shared + Routed Expert 분리 (DeepSeek-V2)
  │ 해결: 범용 지식은 shared, 전문 지식은 routed
  ▼
256개 Fine-grained + Shared Expert (DeepSeek-V3)
  └ 현재 최대 규모
```

### 4.3 학습 안정성 문제와 해결

MoE의 학습 불안정성은 여러 원인에서 기인한다:

1. **라우터 로짓 폭발(Router Logit Explosion):** 학습이 진행됨에 따라 라우터의 로짓 값이 과도하게 커져, softmax 출력이 one-hot에 가까워지고 그래디언트가 소실된다. ST-MoE의 router z-loss가 이를 해결한다.

2. **전문가 붕괴(Expert Collapse):** 소수의 전문가만 반복적으로 선택되어 학습되고, 나머지 전문가는 사장된다. 이는 양의 피드백 루프(positive feedback loop)에 의해 발생한다: 특정 전문가가 자주 선택됨 → 해당 전문가가 더 잘 학습됨 → 더 자주 선택됨.

    **해결 전략:**
    - 보조 부하 균형 손실 (Switch Transformer)
    - 노이즈 주입 (Shazeer et al., 2017)
    - Expert parallelism에서의 jitter noise (GShard)
    - 주기적 전문가 리셋 (Chi et al., 2022)

3. **부동소수점 정밀도 문제:** MoE의 라우터 연산은 수치적으로 민감하여, bfloat16에서 발산할 수 있다. Switch Transformer는 라우터만 float32로 연산하는 selective precision을 제안하였다.

### 4.4 부하 균형 전략의 비교

| 전략 | 방법 | 장점 | 단점 |
|------|------|------|------|
| **Auxiliary Loss** (Switch) | 부하 분산 보조 손실 가산 | 단순, 효과적 | 하이퍼파라미터 민감, 주 목표 간섭 |
| **Capacity Factor** (GShard) | 전문가당 최대 토큰 수 제한 | 메모리 예측 가능 | 토큰 드롭 |
| **Expert Choice** (Zhou) | 전문가가 토큰 선택 | 완벽한 부하 균형 | 자기회귀 제한 |
| **Bias-based** (DeepSeek-V3) | 동적 바이어스 조정 | 보조 손실 불필요, 성능 간섭 없음 | 바이어스 업데이트 규칙 설계 필요 |
| **Sinkhorn Routing** (BASE) | Sinkhorn 알고리즘으로 할당 최적화 | 이론적 최적 | 연산 비용 높음 |

### 4.5 분산 MoE에서의 통신 오버헤드

MoE 모델의 분산 학습에서는 전문가가 서로 다른 디바이스에 배치되므로, 토큰을 해당 전문가의 디바이스로 전송하는 all-to-all 통신이 필수적이다.

**통신량 분석:**

$$
\text{통신량} = 2 \times B \times S \times d \times K / N_{\text{devices}}
$$

여기서 $B$는 배치 크기, $S$는 시퀀스 길이, $d$는 hidden dimension, $K$는 활성 전문가 수이다. Factor 2는 순방향과 역방향 전파를 고려한 것이다.

**해결 전략:**
- **Expert Parallelism + Data Parallelism 혼합:** Lepikhin et al. (2021)
- **Device-Limited Routing:** 토큰이 라우팅되는 디바이스 수를 제한 (DeepSeek-V2)
- **Expert Buffering:** 통신과 연산을 중첩(overlap)하여 파이프라인 효율 극대화 (Tutel, Hwang et al., 2023)
- **MoE Parallelism 전용 프레임워크:** Megablocks (Gale et al., 2023)는 블록 희소 행렬 연산으로 MoE를 효율화한다

### 4.6 MoE 모델의 추론 최적화

MoE 모델은 총 파라미터가 크므로, 추론 시 전체 모델을 메모리에 로드해야 하는 문제가 있다. 이에 대한 해결 전략은 다음과 같다:

1. **Expert Offloading:** 비활성 전문가를 CPU 메모리나 SSD에 오프로드하고, 필요 시에만 GPU로 로드한다. Eliseev & Mazur (2023)는 LRU 캐싱 기반 오프로딩으로 Mixtral 8x7B를 단일 GPU에서 추론하는 방법을 제시하였다.

2. **Expert Pruning:** 활용도가 낮은 전문가를 제거하여 모델 크기를 줄인다. Lu et al. (2024)은 MoE 전문가의 중요도를 측정하고 하위 전문가를 제거하는 체계적 방법론을 제시하였다.

3. **Expert Merging:** 유사한 전문가를 병합하여 전문가 수를 줄인다.

4. **양자화:** MoE 전문가를 개별적으로 양자화한다. 전문가별 활성화 분포가 상이하므로, 전문가별 양자화 파라미터를 독립적으로 설정해야 한다.

### 4.7 MoE와 Dense 모델의 관계

MoE 모델이 Dense 모델과 근본적으로 다른 표현을 학습하는지에 대한 연구가 진행되고 있다. Jiang et al. (2024)은 Mixtral 8x7B의 전문가 활성화 패턴을 분석하여, (1) 전문가는 토픽보다 구문적 역할에 따라 활성화되는 경향이 있으며, (2) 초기 레이어에서는 전문가 선택이 비교적 균일하고, 후기 레이어에서 더 전문화된 패턴이 나타남을 보였다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Adaptive Mixtures of Local Experts | Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton | **Neural Computation 1991** | https://doi.org/10.1162/neco.1991.3.1.79 |
| 2 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, Jeff Dean | **ICLR 2017** | https://arxiv.org/abs/1701.06538 |
| 3 | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen | **ICLR 2021** | https://arxiv.org/abs/2006.16668 |
| 4 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | William Fedus, Barret Zoph, Noam Shazeer | **JMLR 2022** | https://arxiv.org/abs/2101.03961 |
| 5 | ST-MoE: Designing Stable and Transferable Sparse Expert Models | Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus | **arXiv 2022** | https://arxiv.org/abs/2202.08906 |
| 6 | Hash Layers For Large Sparse Models | Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston | **NeurIPS 2021** | https://arxiv.org/abs/2106.04426 |
| 7 | Mixture-of-Experts with Expert Choice Routing | Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc V. Le, James Laudon | **NeurIPS 2022** | https://arxiv.org/abs/2202.09368 |
| 8 | Mixtral of Experts | Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. | **arXiv 2024** | https://arxiv.org/abs/2401.04088 |
| 9 | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, et al. | **ACL 2024** | https://arxiv.org/abs/2401.06066 |
| 10 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek AI | **arXiv 2024** | https://arxiv.org/abs/2405.04434 |
| 11 | DeepSeek-V3 Technical Report | DeepSeek AI | **arXiv 2024** | https://arxiv.org/abs/2412.19437 |
| 12 | Unified Scaling Laws for Routed Language Models | Aidan Clark, Diego de las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. | **ICML 2022** | https://arxiv.org/abs/2202.01169 |
| 13 | Scaling Laws for Neural Language Models | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei | **arXiv 2020** | https://arxiv.org/abs/2001.08361 |
| 14 | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | Yoshua Bengio, Nicholas Leonard, Aaron Courville | **arXiv 2013** | https://arxiv.org/abs/1308.3432 |
| 15 | DBRX: A New State-of-the-Art Open LLM | Databricks Mosaic AI | **Databricks Blog 2024** | https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm |
| 16 | Qwen2 Technical Report | Qwen Team (Alibaba) | **arXiv 2024** | https://arxiv.org/abs/2407.10671 |
| 17 | BASE Layers: Simplifying Training of Large, Sparse Models | Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, Luke Zettlemoyer | **ICML 2021** | https://arxiv.org/abs/2103.16716 |
| 18 | Tutel: Adaptive Mixture-of-Experts at Scale | Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Saber, Jithin Jose, et al. | **MLSys 2023** | https://arxiv.org/abs/2206.03382 |
| 19 | MegaBlocks: Efficient Sparse Training with Mixture-of-Experts | Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia | **MLSys 2023** | https://arxiv.org/abs/2211.15841 |
| 20 | Hierarchical Mixtures of Experts and the EM Algorithm | Michael I. Jordan, Robert A. Jacobs | **Neural Computation 1994** | https://doi.org/10.1162/neco.1994.6.2.181 |
| 21 | Scaling Vision with Sparse Mixture of Experts | Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, Neil Houlsby | **NeurIPS 2021** | https://arxiv.org/abs/2106.05974 |
| 22 | Fast Inference of Mixture-of-Experts Language Models with Offloading | Artyom Eliseev, Denis Mazur | **arXiv 2023** | https://arxiv.org/abs/2312.17238 |
| 23 | Examining Scaling and Transfer of Language Model Architectures for Machine Translation | Biao Zhang, Behrooz Ghorbani, Ankur Bapna, Yong Cheng, Xavier Garcia, Jonathan Clark, Orhan Firat | **ICML 2022** | https://arxiv.org/abs/2202.00528 |
| 24 | Go Wider Instead of Deeper | Fuzhao Xue, Ziji Shi, Futao Wei, Yuxuan Lou, Yong Liu, Yang You | **AAAI 2022** | https://arxiv.org/abs/2107.11817 |
| 25 | Towards Understanding Mixture of Experts in Deep Learning | Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, Yuanzhi Li | **arXiv 2022** | https://arxiv.org/abs/2208.02813 |
| 26 | Scaling Data-Constrained Language Models | Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin Raffel | **NeurIPS 2023** | https://arxiv.org/abs/2305.16264 |
| 27 | Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints | Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, Neil Houlsby | **ICLR 2023** | https://arxiv.org/abs/2212.05055 |
| 28 | From Sparse to Soft Mixtures of Experts | Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Neil Houlsby | **ICLR 2024** | https://arxiv.org/abs/2308.00951 |
| 29 | Mixtral of Experts (8x22B) | Mistral AI | **Mistral Blog 2024** | https://mistral.ai/news/mixtral-8x22b |
| 30 | Scaling Laws for Fine-Grained Mixture of Experts | Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pioro, Michal Krutul, Sebastian Jaszczur, et al. | **arXiv 2024** | https://arxiv.org/abs/2402.07871 |
| 31 | Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture of Experts | Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, Hongsheng Li | **ACL 2024** | https://arxiv.org/abs/2402.14800 |
| 32 | OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models | Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, Yang You | **arXiv 2024** | https://arxiv.org/abs/2402.01739 |
| 33 | Scattered Mixture-of-Experts Implementation | Shawn Tan, Yikang Shen, Rameswar Panda, Aaron Courville | **arXiv 2023** | https://arxiv.org/abs/2311.15907 |
| 34 | Chi et al., Representation Collapse in Training of Mixture-of-Experts Models | Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, Heyan Huang, Furu Wei | **arXiv 2022** | https://arxiv.org/abs/2204.09179 |
| 35 | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation (MoE in Vision-Language) | DeepSeek AI | **arXiv 2024** | https://arxiv.org/abs/2410.13848 |
