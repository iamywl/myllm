# 양자화 (Quantization for Large Language Models)

## 1. 기법의 정의

양자화(quantization)란 신경망의 가중치(weight) 및 활성값(activation)을 고정밀도 부동소수점 표현(FP32, FP16, BF16)에서 저비트 정수 또는 이산 표현으로 사상(mapping)하는 모델 압축 기법이다. 형식적으로 정의하면, 양자화 함수 $Q: \mathbb{R} \rightarrow \mathcal{S}$는 연속 실수값을 유한 이산 집합 $\mathcal{S}$로 변환하는 비가역적 사상이다.

**균일 양자화(Uniform Quantization)** 의 경우, $b$-비트 양자화 함수는 다음과 같이 정의된다:

$$
Q(x) = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z,\; 0,\; 2^b - 1\right)
$$

여기서 $s$는 스케일 팩터(scale factor), $z$는 영점(zero-point), $\lfloor \cdot \rceil$은 최근접 정수 반올림(round-to-nearest) 연산이다. 역양자화(dequantization)는 다음과 같이 수행된다:

$$
\hat{x} = s \cdot (Q(x) - z)
$$

스케일 팩터 $s$는 대칭(symmetric) 양자화의 경우 $s = \frac{\max(|x|)}{2^{b-1} - 1}$로, 비대칭(asymmetric) 양자화의 경우 $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$로 결정된다.

양자화의 핵심 목표는 두 가지이다. 첫째, 모델의 메모리 풋프린트(memory footprint)를 축소하여 제한된 하드웨어 자원에서의 배포를 가능하게 하는 것이다. 둘째, 추론 시 메모리 대역폭(memory bandwidth) 병목을 완화하여 처리량(throughput)과 지연시간(latency)을 개선하는 것이다. LLM의 추론은 연산 제약(compute-bound)이 아닌 메모리 대역폭 제약(memory-bound)이므로, 가중치 비트 수의 감소는 이론적으로 비례적인 처리량 향상을 가져온다.

---

## 2. 기존 기법의 한계와 양자화 등장 배경

### 2.1 부동소수점 학습의 메모리 문제

Transformer 기반 LLM의 규모가 급격히 증가함에 따라 모델 배포의 실용적 장벽이 부각되었다. LLaMA-65B 모델은 FP16 표현 시 약 130GB의 메모리를 요구하며, 이는 단일 NVIDIA A100-80GB GPU의 용량을 초과한다. GPT-3 175B의 경우 FP16으로 약 350GB에 달하여 최소 5장의 A100 GPU가 필요하다. 이러한 메모리 요구량은 서빙 비용을 직접적으로 증가시키며, 소비자 하드웨어에서의 실행을 원천적으로 차단한다.

**문제:** FP32/FP16 표현은 신경망 가중치의 실제 정보 엔트로피에 비해 과도한 비트를 할당하고 있다.

### 2.2 혼합 정밀도 학습의 등장과 한계

Micikevicius et al. (2018)이 제안한 혼합 정밀도 학습(Mixed Precision Training)은 FP16으로 순전파/역전파를 수행하면서 FP32 마스터 가중치를 유지하는 방식으로, 학습 속도를 약 2배 향상시키되 정확도 손실을 방지하였다. NVIDIA Volta 아키텍처의 Tensor Core가 FP16 행렬곱을 하드웨어적으로 가속함으로써 이 접근이 실용화되었다. BF16(Brain Floating Point)은 FP32와 동일한 지수부(exponent) 범위를 유지하면서 가수부(mantissa)를 축소한 16비트 포맷으로, 학습 안정성에서 FP16보다 우수한 것으로 확인되었다.

**문제:** 혼합 정밀도 학습은 주로 학습 효율성에 초점을 맞추며, 추론 시 메모리 절감은 FP32 대비 2배에 불과하다. 이는 수십~수백 GB 규모의 LLM 배포 문제를 근본적으로 해결하지 못한다.

**해결 → 새로운 문제:** 16비트 표현 이하로의 양자화가 필요하나, 나이브(naive)한 INT8/INT4 양자화는 LLM에서 심각한 성능 저하를 유발한다. 특히 LLM의 활성값에서 나타나는 이상치(outlier) 특성이 기존 양자화 기법의 직접 적용을 방해하는 것으로 밝혀졌다.

### 2.3 PTQ와 QAT의 구분

양자화 기법은 적용 시점에 따라 크게 두 범주로 분류된다.

**사후 학습 양자화(Post-Training Quantization, PTQ)** 는 이미 학습이 완료된 모델에 양자화를 적용하는 방식이다. 소량의 캘리브레이션 데이터(보통 128~1024 샘플)만으로 양자화 파라미터를 결정하므로 연산 비용이 낮다. 그러나 양자화 오류에 대한 보상이 제한적이다.

**양자화 인식 학습(Quantization-Aware Training, QAT)** 은 학습 과정에서 양자화 연산을 시뮬레이션하여 모델이 양자화 오류에 적응하도록 하는 방식이다. Straight-Through Estimator(STE)를 사용하여 불연속적인 양자화 함수의 기울기를 근사한다:

$$
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial Q(x)} \cdot \mathbf{1}_{x \in [x_{\min}, x_{\max}]}
$$

QAT는 PTQ보다 우수한 정확도를 달성하지만, LLM 규모에서 전체 재학습 비용이 막대하여(수십만 달러 이상) 실용적 적용이 어렵다. 이로 인해 LLM 양자화 연구는 PTQ 중심으로 발전하였다.

| 속성 | PTQ | QAT |
|------|-----|-----|
| **적용 시점** | 학습 완료 후 | 학습 과정 중 |
| **연산 비용** | 낮음 (수 시간 이내) | 극히 높음 (전체 학습의 수십 %) |
| **캘리브레이션 데이터** | 128~1024 샘플 | 전체 학습 데이터셋 |
| **정확도 (4-bit)** | 양호~우수 | 우수~최고 |
| **대표 기법** | GPTQ, AWQ, SmoothQuant | LLM-QAT, BitNet, OneBit |
| **실용성** | 높음 | 낮음 (학습 인프라 필요) |

---

## 3. 주요 기법 상세

### 3.1 Mixed Precision Training (Micikevicius et al., 2018)

**문제:** FP32 학습은 메모리 사용량과 연산 비용이 과도하나, 단순 FP16 전환은 수치적 불안정성(gradient underflow, overflow)을 유발한다.

**해결:** 혼합 정밀도 학습은 세 가지 기법을 결합한다: (1) FP32 마스터 가중치 유지, (2) 손실 스케일링(loss scaling)을 통한 gradient underflow 방지, (3) FP16 행렬곱 후 FP32 누적. 이 접근은 NVIDIA Tensor Core를 활용하여 학습 속도를 1.5~3배 향상시키되, ImageNet top-1 정확도 차이를 0.1% 이내로 유지하였다.

**새로운 문제:** 학습 효율성은 개선되었으나, 추론 배포 시 모델 크기 자체의 근본적 축소가 이루어지지 않는다. 16비트 이하의 공격적 양자화가 필요하게 되었다.

### 3.2 LLM.int8() (Dettmers et al., NeurIPS 2022)

**문제:** LLM을 단순 INT8로 양자화하면 특정 은닉 차원(hidden dimension)에서 나타나는 **이상치 특성(emergent outlier features)** 으로 인해 심각한 성능 저하가 발생한다. Dettmers et al.은 6.7B 파라미터 이상의 모델에서 소수(약 0.1%)의 은닉 차원이 나머지 차원보다 100배 이상 큰 활성값을 생성하는 현상을 발견하였다. 이러한 이상치를 INT8 범위로 압축하면 나머지 99.9% 차원의 양자화 해상도가 극도로 저하된다.

**해결:** 분해 혼합 정밀도(mixed-precision decomposition) 기법을 제안하였다. 행렬곱 $\mathbf{Y} = \mathbf{X}\mathbf{W}$에서 이상치 차원 집합 $\mathcal{O}$를 식별하고, 행렬을 두 부분으로 분리한다:

$$
\mathbf{Y} = \mathbf{X}_{\mathcal{O}} \mathbf{W}_{\mathcal{O}} + \text{Int8}(\mathbf{X}_{\bar{\mathcal{O}}}) \cdot \text{Int8}(\mathbf{W}_{\bar{\mathcal{O}}})
$$

이상치 차원($|\mathcal{O}|$은 전체의 약 0.1%)은 FP16으로 처리하고, 나머지는 벡터 단위(vector-wise) INT8 양자화를 적용한다. 이 방식으로 OPT-175B, BLOOM-176B 등 최대 규모 모델에서 **성능 저하 없이** INT8 추론이 가능함을 입증하였다. OPT-175B에서 FP16 대비 perplexity 차이가 0.1 이내였다.

**실험 결과:**
- OPT-175B: FP16 perplexity 8.34 → LLM.int8() perplexity 8.35 (WikiText-2)
- BLOOM-176B: FP16 대비 perplexity 증가 < 0.1
- 메모리 절감: FP16 대비 약 2배

**새로운 문제:** INT8은 메모리를 2배만 절감하며, 더 공격적인 4비트 양자화가 필요하다. 또한 이상치 차원의 FP16 처리로 인해 INT8 전용 가속기의 이점을 완전히 활용하지 못한다.

### 3.3 SmoothQuant (Xiao et al., ICML 2023)

**문제:** LLM의 활성값(activation)은 채널별 이상치로 인해 가중치보다 양자화가 현저히 어렵다. 가중치-활성값 동시 양자화(W8A8)에서 활성값 양자화가 병목이 된다.

**해결:** SmoothQuant는 양자화 난이도를 활성값에서 가중치로 이전(migrate)하는 수학적으로 동등한 변환을 제안하였다. 선형 레이어 $\mathbf{Y} = \mathbf{X}\mathbf{W}$에 채널별 스케일링 인자 $\mathbf{s}$를 도입한다:

$$
\mathbf{Y} = (\mathbf{X} \text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s}) \mathbf{W}) = \hat{\mathbf{X}} \hat{\mathbf{W}}
$$

여기서 스케일링 인자는 활성값과 가중치의 최대값 비율로 결정된다:

$$
s_j = \frac{\max(|\mathbf{X}_j|)^\alpha}{\max(|\mathbf{W}_j|)^{1-\alpha}}
$$

하이퍼파라미터 $\alpha \in [0, 1]$는 양자화 난이도 분배를 제어하며, 실험적으로 $\alpha = 0.5$가 대부분의 모델에서 최적임을 확인하였다. 이 변환 후 $\hat{\mathbf{X}}$의 이상치가 완화되어 W8A8 양자화가 가능해진다.

**실험 결과:**
- OPT-175B (W8A8): FP16 perplexity 8.34 → SmoothQuant perplexity 8.42 (WikiText-2)
- LLaMA-65B (W8A8): FP16 대비 perplexity 증가 < 0.3
- 추론 속도: FP16 대비 약 1.56배 향상 (NVIDIA INT8 Tensor Core 활용)

**새로운 문제:** W8A8은 메모리 절감이 2배에 그치며, 4비트 가중치 양자화에 비해 압축률이 부족하다. 활성값 양자화 없이 가중치만 4비트로 양자화하는 접근(W4A16)이 실용적 관점에서 더 매력적인 것으로 확인되었다.

### 3.4 GPTQ: OBS 프레임워크 기반 4비트 양자화 (Frantar et al., ICLR 2023)

**문제:** 나이브한 라운드-투-니어리스트(RTN, round-to-nearest) 4비트 양자화는 LLM에서 심각한 perplexity 저하를 유발한다. OPT-175B의 경우 RTN 4비트 양자화 시 perplexity가 8.34에서 14.89로 급증한다.

**해결:** GPTQ는 Optimal Brain Surgeon(OBS) 프레임워크를 LLM 규모로 확장한 기법이다. OBS는 가중치를 제거(또는 양자화)할 때 발생하는 목적함수 변화를 Hessian 행렬의 역행렬을 통해 2차 근사하는 프레임워크이다.

레이어의 가중치 행렬 $\mathbf{W}$에 대해, 양자화 오류를 최소화하는 문제를 다음과 같이 정식화한다:

$$
\underset{\hat{\mathbf{W}}}{\arg\min} \; \| \mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X} \|_2^2
$$

이는 등가적으로 각 행에 대해 다음 문제로 분해된다:

$$
\underset{\hat{\mathbf{w}}}{\arg\min} \; (\mathbf{w} - \hat{\mathbf{w}})^T \mathbf{H} (\mathbf{w} - \hat{\mathbf{w}})
$$

여기서 $\mathbf{H} = 2\mathbf{X}\mathbf{X}^T$는 Hessian 행렬이다. OBS 프레임워크에 따르면, 가중치 $w_q$를 양자화할 때의 최적 보상은 다음과 같다:

$$
\boldsymbol{\delta}_{\mathcal{F}} = -\frac{w_q - \text{quant}(w_q)}{[\mathbf{H}_{\mathcal{F}}^{-1}]_{qq}} \cdot (\mathbf{H}_{\mathcal{F}}^{-1})_{:,q}
$$

여기서 $\mathcal{F}$는 아직 양자화되지 않은 가중치 집합이며, $(\mathbf{H}_{\mathcal{F}}^{-1})_{:,q}$는 역Hessian의 $q$-번째 열이다.

GPTQ의 핵심 기여는 세 가지이다: (1) 모든 행에 동일한 양자화 순서를 적용하여 행렬 연산으로 배치 처리 가능하게 함 (임의 순서 대비 정확도 손실이 미미함을 실험적으로 입증), (2) Cholesky 분해를 이용한 효율적 역Hessian 갱신, (3) 128열 단위의 블록 양자화로 메모리-연산 효율성 확보. 이를 통해 OPT-175B를 약 4시간 만에 4비트로 양자화할 수 있었다.

**실험 결과 (WikiText-2 perplexity):**
- OPT-175B: FP16 8.34 → GPTQ-4bit 8.68 (RTN-4bit 14.89 대비 극적 개선)
- LLaMA-65B: FP16 3.53 → GPTQ-4bit 3.84
- BLOOM-176B: FP16 8.11 → GPTQ-4bit 8.43
- LLaMA-30B: FP16 4.10 → GPTQ-3bit 5.69, GPTQ-4bit 4.23

**새로운 문제:** GPTQ는 양자화 과정에서 캘리브레이션 데이터에 의존하며, 양자화 시간이 수 시간에 달한다(65B 모델 기준). 또한 모든 가중치를 동등하게 취급하여, 성능에 결정적인 소수 가중치의 중요도를 반영하지 못한다. GPU 전용 커널이 필요하며 CPU 추론은 지원하지 않는다.

### 3.5 AWQ: 활성값 인식 가중치 양자화 (Lin et al., MLSys 2024)

**문제:** GPTQ를 포함한 기존 기법들은 가중치의 중요도 차이를 충분히 고려하지 않는다. 가중치의 극히 일부만이 모델 성능에 결정적이라는 관찰이 활용되지 않고 있었다.

**해결:** AWQ는 가중치 채널의 중요도를 해당 채널의 **활성값(activation) 크기** 로 판별하는 핵심 관찰에 기반한다. 입력 활성값의 평균 크기가 큰 채널의 가중치가 모델 출력에 더 큰 영향을 미치므로, 이러한 채널의 양자화 오류를 우선적으로 최소화해야 한다.

구체적으로, 채널별 스케일링 인자 $\mathbf{s}$를 도입하여 가중치를 변환한다:

$$
Q(\mathbf{w} \cdot s) \cdot \frac{\mathbf{x}}{s} \approx \mathbf{w} \cdot \mathbf{x}
$$

스케일링 인자 $s > 1$은 중요 채널의 양자화 해상도를 높이고(상대적 양자화 오류 $\Delta(w \cdot s) / (w \cdot s)$가 감소), $s < 1$은 비중요 채널의 해상도를 낮추는 효과를 가진다. 최적 스케일링 인자는 다음 목적함수를 최소화하도록 탐색된다:

$$
\mathbf{s}^* = \underset{\mathbf{s}}{\arg\min} \; \| Q(\mathbf{W} \cdot \text{diag}(\mathbf{s})) (\text{diag}(\mathbf{s})^{-1} \mathbf{X}) - \mathbf{W}\mathbf{X} \|
$$

실험적으로 $s_j = (\bar{|X_j|})^\alpha$에서 $\alpha$를 그리드 서치하는 것이 효과적이며, 전형적으로 $\alpha \in [0, 1]$의 범위에서 최적값이 결정된다.

**실험 결과 (WikiText-2 perplexity):**
- LLaMA-7B: FP16 5.68 → AWQ-4bit 5.78 (GPTQ-4bit 5.85)
- LLaMA-13B: FP16 5.09 → AWQ-4bit 5.19 (GPTQ-4bit 5.20)
- LLaMA-70B: FP16 3.32 → AWQ-4bit 3.41
- 양자화 속도: GPTQ 대비 약 10~100배 빠름 (역Hessian 계산 불필요)

**새로운 문제:** AWQ는 GPU 커널에 최적화되어 있으며, CPU 추론 환경에서의 지원이 부족하다. 또한 4비트 이하의 극한 양자화(2~3비트)에서의 성능이 검증되지 않았다.

### 3.6 GGML/GGUF 포맷과 K-Quant (Gerganov, 2023)

**문제:** GPTQ와 AWQ는 GPU 전용으로 설계되어, GPU가 없는 소비자 하드웨어(CPU, Apple Silicon)에서의 LLM 추론이 불가능하다.

**해결:** Georgi Gerganov가 개발한 llama.cpp 프로젝트는 순수 C/C++ 구현으로 CPU에서의 LLM 추론을 가능하게 하였다. GGML(GPT-Generated Model Language) 및 이후 GGUF(GPT-Generated Unified Format) 포맷은 양자화된 텐서와 메타데이터를 단일 파일로 패키징하는 표준을 제시하였다.

K-Quant 양자화 체계는 레이어의 중요도에 따라 비트 수를 차등 할당하는 혼합 양자화(mixed quantization)를 적용한다. 양자화 수준은 Q2_K(평균 약 2.6비트), Q3_K_S/M/L(약 3.0~3.9비트), Q4_K_S/M(약 4.0~4.5비트), Q5_K_S/M(약 5.0~5.5비트), Q6_K(약 6.5비트), Q8_0(8비트)으로 구분된다. 각 레이어는 Attention 레이어와 FFN 레이어의 중요도를 고려하여 서로 다른 비트 수가 할당된다.

기술적으로 K-Quant는 슈퍼블록(super-block) 구조를 사용한다. 예를 들어, Q4_K_M은 256개의 가중치를 하나의 슈퍼블록으로 묶고, 이를 다시 8개의 서브블록(각 32개 가중치)으로 분할하여 서브블록별 4비트 양자화와 6비트 최소값을 적용한다.

**실험 결과 (LLaMA-2-7B, WikiText-2 perplexity):**
- FP16: 5.47 → Q8_0: 5.48 → Q6_K: 5.53 → Q5_K_M: 5.69 → Q4_K_M: 5.90 → Q3_K_M: 6.58 → Q2_K: 9.08

**새로운 문제:** 2~3비트 양자화에서 perplexity 저하가 급격하며, CPU 추론 속도는 GPU 대비 현저히 느리다. 또한 GGUF 양자화는 이론적 최적성이 보장되지 않는 경험적(heuristic) 접근이다.

### 3.7 SqueezeLLM: 밀집-희소 양자화 (Kim et al., ICML 2024)

**문제:** 균일 양자화(uniform quantization)는 가중치 분포의 비균일성을 반영하지 못한다. LLM 가중치 분포는 정규분포에 가깝지만, 소수의 이상치(outlier)가 양자화 범위를 지배하여 대다수 가중치의 양자화 해상도가 저하된다.

**해결:** SqueezeLLM은 두 가지 핵심 기법을 결합한다.

첫째, **감도 기반 비균일 양자화(sensitivity-based non-uniform quantization)** 이다. 가중치의 양자화 감도를 Fisher 정보 행렬의 대각 근사로 측정하고, 감도가 높은 영역에 더 많은 양자화 수준을 배분한다. 양자화 수준 $\{c_1, c_2, \ldots, c_{2^b}\}$을 학습하여 가중치별 감도 가중 양자화 오류를 최소화한다:

$$
\min_{\{c_k\}} \sum_i F_{ii} \cdot (w_i - c_{q(w_i)})^2
$$

여기서 $F_{ii}$는 Fisher 정보 행렬의 $i$-번째 대각 원소이다.

둘째, **밀집-희소 분해(Dense-and-Sparse decomposition)** 이다. 양자화에 민감한 이상치 가중치를 희소 행렬로 분리 저장하고, 나머지를 밀집(dense) 양자화 행렬로 표현한다. 이를 통해 이상치가 양자화 범위에 미치는 영향을 제거한다.

**실험 결과 (WikiText-2 perplexity):**
- LLaMA-7B: GPTQ-3bit 6.61 → SqueezeLLM-3bit 6.18
- LLaMA-13B: GPTQ-3bit 5.36 → SqueezeLLM-3bit 5.18
- LLaMA-7B 4bit: GPTQ 5.85 → SqueezeLLM 5.75

**새로운 문제:** 비균일 양자화는 전용 커널 구현이 필요하며, 희소 행렬 연산의 하드웨어 가속이 제한적이다. 또한 Fisher 정보 행렬 계산에 추가 비용이 소요된다.

### 3.8 SpQR: 희소-양자화 표현 (Dettmers et al., ICLR 2024)

**문제:** 가중치의 소수 이상치가 전체 양자화 오류를 지배하나, 이를 별도 처리하는 체계적 프레임워크가 부재하였다.

**해결:** SpQR(Sparse-Quantized Representation)은 가중치를 민감도에 따라 이중 표현(dual representation)으로 저장한다. 각 가중치의 양자화 민감도를 다음과 같이 측정한다:

$$
\text{sensitivity}(w_{ij}) = \frac{(w_{ij} - Q(w_{ij}))^2}{[\mathbf{H}^{-1}]_{jj}}
$$

이 감도가 임계값을 초과하는 가중치(전체의 약 1~2%)는 고정밀도(FP16)로 유지하고, 나머지는 3비트 양자화를 적용한다. 이상치 인덱스는 희소 행렬 포맷(CSR)으로 저장하며, 평균 유효 비트 수는 약 3.01~3.86비트이다.

**실험 결과 (WikiText-2 perplexity):**
- LLaMA-65B: FP16 3.53 → SpQR-3bit(평균 3.14비트) 3.89
- LLaMA-65B: SpQR-3bit < GPTQ-4bit에 근접하면서 평균 비트 수 약 20% 절감

**새로운 문제:** 희소 표현의 비정형적 메모리 접근 패턴이 GPU 캐시 효율을 저하시키며, 실제 추론 속도 향상이 이론적 압축률 개선에 미치지 못한다.

### 3.9 QuIP 및 QuIP#: 비간섭성 처리와 격자 코드북 (Chee et al., ICML 2024)

**문제:** 2비트 양자화에서 기존 기법들의 perplexity 저하가 과도하다. GPTQ 2비트 양자화 시 LLaMA-2-7B의 perplexity가 50 이상으로 폭증하여 사실상 사용 불가능하다.

**해결 (QuIP):** Chee et al.은 양자화 오류의 상한(upper bound)이 Hessian 행렬의 비간섭성(incoherence)에 의존함을 이론적으로 증명하고, 직교 변환(orthogonal transform)으로 비간섭성을 극대화하는 LDLQ(LDL Quantization) 알고리즘을 제안하였다. 구체적으로, 가중치 행렬 $\mathbf{W}$와 Hessian $\mathbf{H}$에 랜덤 직교 행렬 $\mathbf{U}, \mathbf{V}$를 적용한다:

$$
\tilde{\mathbf{W}} = \mathbf{U}\mathbf{W}\mathbf{V}, \quad \tilde{\mathbf{H}} = \mathbf{V}^T \mathbf{H} \mathbf{V}
$$

이 변환은 가중치와 Hessian의 원소를 균일하게 분산시켜 양자화 친화적으로 만든다.

**해결 (QuIP#):** 후속 연구인 QuIP#은 두 가지 핵심 개선을 도입하였다. 첫째, 랜덤 직교 행렬 대신 **랜덤화된 Hadamard 변환(randomized Hadamard transform)** 을 사용하여 $O(n \log n)$ 복잡도로 비간섭성 처리를 수행한다. Hadamard 행렬 $\mathbf{H}_n$은 $\mathbf{H}_1 = [1]$에서 크로네커 곱으로 재귀 구성된다:

$$
\mathbf{H}_{2n} = \frac{1}{\sqrt{2}} \begin{bmatrix} \mathbf{H}_n & \mathbf{H}_n \\ \mathbf{H}_n & -\mathbf{H}_n \end{bmatrix}
$$

둘째, 스칼라 양자화 대신 **격자 코드북(lattice codebook)** 기반 벡터 양자화를 적용한다. $E_8$ 격자(8차원 격자)의 양자화 효율(quantization efficiency)은 이론적 최적에 근접하며, 8개 가중치를 하나의 벡터로 묶어 격자점으로 양자화한다. $E_8$ 격자의 코드북은 다음과 같이 정의된다:

$$
E_8 = \left\{ \mathbf{x} \in \mathbb{Z}^8 \cup (\mathbb{Z} + \tfrac{1}{2})^8 : \sum_i x_i \in 2\mathbb{Z},\; \|\mathbf{x}\|^2 \leq r^2 \right\}
$$

2비트 양자화(벡터당 16비트, 원소당 2비트)에서 $E_8$ 격자의 정규화 이차 모멘트(normalized second moment, NSM)는 약 0.0717로, 스칼라 균일 양자화의 NSM 0.0833 대비 약 14% 우수하다.

**실험 결과 (WikiText-2 perplexity):**
- LLaMA-2-7B: FP16 5.47 → QuIP#-2bit 6.15 (GPTQ-2bit > 50, AQLM-2bit 6.81)
- LLaMA-2-70B: FP16 3.32 → QuIP#-2bit 4.16
- LLaMA-2-7B: QuIP#-4bit 5.54 (FP16 5.47과 0.07 차이)

**새로운 문제:** $E_8$ 격자 복호화(decoding)에 연산 비용이 소요되며, 추론 커널 최적화가 GPTQ/AWQ 대비 미성숙하다. 또한 벡터 양자화의 병렬화가 스칼라 양자화보다 복잡하다.

### 3.10 AQLM: 가산적 양자화 (Egiazarian et al., ICML 2024)

**문제:** 기존 벡터 양자화 기법들은 단일 코드북의 한계로 인해 2비트에서 충분한 표현력을 확보하기 어렵다.

**해결:** AQLM(Additive Quantization for Language Models)은 다중 코드북 가산적 양자화(multi-codebook additive quantization)를 LLM에 적용한다. 각 가중치 벡터 $\mathbf{w}$를 $M$개 코드북의 코드워드 합으로 근사한다:

$$
\hat{\mathbf{w}} = \sum_{m=1}^{M} \mathbf{c}_{m, i_m}
$$

여기서 $\mathbf{c}_{m, i_m}$은 $m$-번째 코드북의 $i_m$-번째 코드워드이다. 코드북 학습과 코드 할당은 교대 최적화(alternating optimization)로 수행되며, beam search를 통해 레이어별 양자화 오류를 최소화한다. 추가적으로 미세조정(fine-tuning) 단계를 도입하여 잔차 오류를 보상한다.

2비트 양자화의 경우 그룹 크기 8, 코드북 2개(각 $2^8 = 256$ 코드워드)를 사용하면 가중치당 평균 $\frac{2 \times 8}{8} = 2$비트가 된다.

**실험 결과 (WikiText-2 perplexity):**
- LLaMA-2-7B: FP16 5.47 → AQLM-2bit 6.81 → AQLM-2bit(+fine-tuning) 5.94
- LLaMA-2-70B: FP16 3.32 → AQLM-2bit 4.24
- LLaMA-2-7B: AQLM-3bit 5.71 (FP16에 근접)

**새로운 문제:** 다중 코드북의 추론 시 복호화 비용이 존재하며, 미세조정 단계는 추가적인 GPU 시간을 요구한다. 또한 코드북 메모리 오버헤드가 소형 모델에서는 무시할 수 없다.

### 3.11 EXL2: 혼합 비트 양자화 엔진 (turboderp, 2023)

**문제:** GPTQ의 고정 비트 양자화는 레이어별 양자화 민감도 차이를 반영하지 못한다. 일부 레이어는 2비트로도 충분하나, 다른 레이어는 6비트 이상이 필요하다.

**해결:** EXL2는 ExLlamaV2 추론 엔진의 양자화 포맷으로, 레이어별로 서로 다른 비트 수를 할당하는 혼합 비트 양자화(mixed-bit quantization)를 구현한다. 목표 평균 비트 수(예: 3.5비트)가 주어지면, 레이어별 양자화 오류를 측정하고 오류가 가장 큰 레이어에 우선적으로 높은 비트를 할당하는 탐욕적(greedy) 알고리즘을 사용한다.

각 레이어는 {2, 3, 4, 5, 6, 8}비트 중 하나로 양자화되며, 동일 레이어 내에서도 서브그룹별로 비트 수가 다를 수 있다. 이는 사실상 연속적인 비트폭 조절을 가능하게 한다 (예: 평균 3.25비트, 4.65비트 등).

**실험 결과 (WikiText-2 perplexity, LLaMA-2-7B 기준):**
- EXL2 4.0bpw: ~5.85 (GPTQ-4bit와 유사)
- EXL2 3.0bpw: ~6.55 (GPTQ-3bit 6.61 대비 개선)
- EXL2 2.5bpw: ~8.10

**새로운 문제:** ExLlamaV2 엔진 전용이므로 생태계 호환성이 제한적이다. 또한 최적 비트 배분 알고리즘이 탐욕적 접근에 의존하여 전역 최적성이 보장되지 않는다.

### 3.12 BitNet 및 BitNet b1.58: 1비트 LLM (Wang et al., 2023; Ma et al., 2024)

**문제:** 기존 PTQ 기법들은 사전학습된 모델의 가중치를 사후적으로 압축하므로, 양자화 오류가 불가피하다. 처음부터 극도로 낮은 비트로 학습할 수 있다면 이 문제를 우회할 수 있다.

**해결 (BitNet):** Wang et al.은 Transformer의 선형 레이어를 BitLinear로 대체하여 가중치를 이진값({-1, +1})으로 제한하는 1비트 LLM 아키텍처를 제안하였다. 학습 시 STE를 사용하여 이진 양자화를 통과하는 기울기를 전파한다:

$$
\tilde{w}_{ij} = \text{Sign}(w_{ij}) = \begin{cases} +1 & \text{if } w_{ij} > 0 \\ -1 & \text{otherwise} \end{cases}
$$

활성값은 absmax 양자화로 $b$비트(보통 8비트)로 양자화된다:

$$
\tilde{x} = \text{Quant}(x) = \text{Clip}\left(\left\lfloor \frac{x}{Q_b} \right\rceil \cdot Q_b, \; -Q_b, \; Q_b\right), \quad Q_b = \frac{\gamma}{2^{b-1}}
$$

여기서 $\gamma = \max(|\mathbf{x}|)$이다.

**해결 (BitNet b1.58):** Ma et al.은 가중치를 삼진값(ternary) {-1, 0, +1}로 확장한 BitNet b1.58을 제안하였다. "b1.58"은 $\log_2(3) \approx 1.58$비트에서 유래한다. 양자화 함수는 다음과 같다:

$$
\tilde{w}_{ij} = \text{RoundClip}\left(\frac{w_{ij}}{\bar{\gamma}}, -1, 1\right), \quad \bar{\gamma} = \frac{1}{nm} \sum_{ij} |w_{ij}|
$$

0값의 도입은 두 가지 이점을 제공한다. 첫째, 특성 필터링(feature filtering) 효과로 특정 입력 차원을 선택적으로 무시할 수 있어 표현력이 향상된다. 둘째, 행렬곱이 덧셈과 뺄셈만으로 수행되어 곱셈 연산이 완전히 제거된다:

$$
y = \tilde{\mathbf{W}} \mathbf{x} = \sum_j \tilde{w}_j x_j = \sum_{j: \tilde{w}_j=1} x_j - \sum_{j: \tilde{w}_j=-1} x_j
$$

이는 에너지 효율과 연산 속도 측면에서 근본적인 이점을 제공한다.

**실험 결과:**
- BitNet b1.58 3B vs LLaMA-3B (FP16): WikiText-2 perplexity가 동등 (약 8.5)
- BitNet b1.58 3.9B vs LLaMA-3B: perplexity, 다운스트림 벤치마크에서 FP16 모델과 동등~우수
- BitNet b1.58 70B: FP16 LLaMA-70B 대비 메모리 11.4배 절감, 에너지 소비 71.4배 절감 (이론적)
- 추론 지연시간: FP16 대비 2.71배 빠름 (3.9B 모델 기준)

**새로운 문제:** QAT 방식이므로 처음부터 학습해야 하며, 기존 사전학습된 모델의 변환이 불가능하다. 현재 범용 하드웨어에서의 1.58비트 연산 가속이 지원되지 않으며, 전용 하드웨어 또는 커널 개발이 필수적이다. 또한 대규모(70B+) 학습의 안정성이 충분히 검증되지 않았다.

### 3.13 NormalFloat 4-bit (NF4) 양자화 (Dettmers et al., NeurIPS 2023)

QLoRA 논문에서 도입된 NF4(NormalFloat 4-bit)는 사전학습된 신경망 가중치가 근사적으로 정규분포 $\mathcal{N}(0, \sigma^2)$를 따른다는 관찰에 기반한 정보 이론적 최적 데이터 타입이다. NF4의 $2^b = 16$개 양자화 수준 $\{q_i\}$는 다음 조건을 만족하도록 설계된다:

$$
P(q_i \leq x < q_{i+1}) = \frac{1}{2^b}, \quad \forall i
$$

즉, 정규분포의 분위수(quantile)를 양자화 수준으로 사용하여 각 양자화 구간에 동일한 확률 질량을 배분한다. 이는 정보 이론적으로 정규분포 데이터에 대한 최적 양자화임이 증명 가능하다. NF4는 QLoRA에서 기저 모델의 4비트 양자화에 사용되며, 일반적인 INT4/FP4 대비 우수한 정확도를 보인다.

**실험 결과:**
- LLaMA-65B: NF4 양자화 + QLoRA 미세조정이 FP16 전체 미세조정의 99.3% 성능 달성
- NF4 vs FP4 vs INT4: NF4가 perplexity 기준으로 일관되게 최우수

### 3.14 QAT의 최신 발전: LLM-QAT 및 관련 연구

**문제:** PTQ는 비용 효율적이나, 4비트 이하에서 성능 저하가 불가피하다. QAT는 이론적으로 우수하나 LLM 규모에서의 비용이 장벽이다.

**해결 (LLM-QAT, Liu et al., 2024):** LLM의 QAT 비용을 줄이기 위해 데이터 프리(data-free) 증류 기법을 제안하였다. 사전학습된 모델 자체의 출력을 학습 데이터로 활용하여 양자화 인식 학습을 수행한다. 가중치와 KV 캐시를 동시에 4비트로 양자화하되, 학습 과정에서 양자화 오류에 대한 적응이 이루어진다.

**실험 결과:**
- LLaMA-7B (W4A8): PTQ(GPTQ) 5.85 → LLM-QAT 5.72 (FP16 5.68에 근접)
- LLaMA-30B (W4A8): LLM-QAT가 GPTQ 대비 perplexity 0.2~0.5 개선

**해결 (EdgeQAT, Chen et al., 2024):** 엣지 디바이스 배포를 위한 경량 QAT로, token-level adaptation과 module-smooth을 결합하여 4비트 양자화의 정확도를 향상시키면서 학습 비용을 절감하였다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 제1단계: 정밀도 축소의 시작 (2017-2021)

혼합 정밀도 학습(Micikevicius et al., 2018)이 FP16 학습의 실용성을 입증하였고, NVIDIA Tensor Core의 하드웨어 지원으로 산업 표준이 되었다. 그러나 이는 학습 효율화에 국한되었으며, 추론 배포 시 모델 크기 축소 문제는 미해결로 남았다. 이 시기의 양자화 연구는 주로 CNN 및 소규모 모델에 집중되었으며(Krishnamoorthi, 2018; Nagel et al., 2019), LLM 특유의 이상치 문제가 인식되지 않았다.

### 4.2 제2단계: LLM 양자화의 개막 (2022)

**LLM.int8() (Dettmers et al., 2022)** 이 LLM에서의 이상치 특성(emergent outlier features)을 최초로 체계적으로 분석하고, 혼합 분해(mixed decomposition) 기법으로 175B 규모에서의 무손실 INT8 양자화를 달성하였다. 이 연구는 두 가지 중요한 사실을 확립하였다: (1) LLM의 활성값 이상치가 양자화의 핵심 장애물이라는 것, (2) 이상치를 별도 처리하면 나머지의 양자화는 용이하다는 것.

→ **인과적 전이:** INT8은 메모리 2배 절감에 그치므로, 4비트 양자화 연구가 촉발되었다. 동시에 이상치 처리의 필요성이 후속 연구(SmoothQuant, SpQR)의 기반이 되었다.

### 4.3 제3단계: PTQ 4비트 양자화의 성숙 (2022-2023)

**GPTQ (Frantar et al., 2023)** 가 OBS 프레임워크를 LLM 규모로 확장하여 4비트 PTQ의 실용성을 입증하였다. Hessian 기반 최적 보상으로 RTN 대비 극적인 perplexity 개선을 달성하였으나, 양자화 속도와 모든 가중치의 동등한 취급이 한계로 지적되었다.

**SmoothQuant (Xiao et al., 2023)** 는 활성값의 양자화 난이도를 가중치로 이전하는 수학적으로 동등한 변환을 제안하여 W8A8 양자화를 가능하게 하였다. 이는 가중치-활성값 동시 양자화라는 별도의 연구 방향을 개척하였다.

**AWQ (Lin et al., 2024)** 는 GPTQ의 한계를 직접적으로 해결하였다. 활성값 크기 기반 가중치 중요도 판별이라는 단순하면서도 효과적인 관찰로, GPTQ 이상의 정확도를 10~100배 빠른 속도로 달성하였다.

→ **인과적 전이:** GPU 전용 4비트 양자화가 성숙하면서, CPU 추론과 극한 양자화(2~3비트)라는 두 방향으로 연구가 분기하였다.

### 4.4 제4단계: CPU 접근성과 민주화 (2023)

**GGML/GGUF (Gerganov, 2023)** 와 llama.cpp 프로젝트는 학술 연구가 아닌 오픈소스 공학적 접근으로, CPU 환경에서의 양자화 LLM 추론을 대중화하였다. K-Quant 체계의 레이어별 차등 비트 할당은 이론적 최적성보다 실용적 효과를 우선시한 경험적 접근이나, 소비자 하드웨어에서의 LLM 접근성을 획기적으로 향상시켰다.

**EXL2 (turboderp, 2023)** 는 유사한 혼합 비트 개념을 GPU 추론에 적용하여, 연속적 비트폭 조절을 가능하게 하였다.

→ **인과적 전이:** 실용적 양자화의 보급이 2비트 양자화의 실용적 필요성을 부각시켰다. 소비자 GPU(8~24GB VRAM)에서 70B 모델을 구동하려면 2~3비트가 필수적이다.

### 4.5 제5단계: 극한 양자화 (2023-2024)

**SqueezeLLM (Kim et al., 2024)** 과 **SpQR (Dettmers et al., 2024)** 가 이상치의 별도 처리(희소 표현)와 비균일 양자화를 결합하여 3비트 양자화의 품질을 개선하였다. 이들은 "밀집 + 희소" 이중 표현이라는 공통 원리를 공유한다.

**QuIP/QuIP# (Chee et al., 2023-2024)** 는 비간섭성 처리(incoherence processing)라는 이론적으로 근거 있는 접근으로 2비트 양자화의 돌파구를 마련하였다. Hadamard 변환과 $E_8$ 격자 코드북의 결합은 2비트에서 이전에 불가능했던 수준의 perplexity를 달성하였다.

**AQLM (Egiazarian et al., 2024)** 은 가산적 벡터 양자화로 다중 코드북의 합으로 가중치를 근사하여 2비트에서 경쟁력 있는 성능을 보였다.

→ **인과적 전이:** PTQ의 이론적 한계(1비트 이하 불가능)가 명확해지면서, 처음부터 저비트로 학습하는 QAT 방향으로의 패러다임 전환이 시작되었다.

### 4.6 제6단계: 1비트 패러다임 전환 (2024-)

**BitNet (Wang et al., 2023)** 과 **BitNet b1.58 (Ma et al., 2024)** 은 가중치를 {-1, 0, +1}로 제한한 QAT 접근으로, 곱셈 연산의 완전한 제거라는 근본적 하드웨어 효율화를 달성하였다. 이는 PTQ 양자화와 질적으로 다른 접근으로, 전용 하드웨어(1-bit accelerator) 설계와 직접 연결된다.

**현재 미해결 문제:** (1) BitNet의 대규모 학습 안정성 및 스케일링 법칙 검증, (2) 1.58비트 전용 하드웨어의 실용화, (3) 기존 사전학습 모델의 1비트 변환 가능성, (4) 극한 양자화에서의 emergent ability 보존 여부.

### 4.7 종합 비교

| 기법 | 비트 | 유형 | LLaMA-2-7B PPL (WikiText-2) | 압축률 | 추론 환경 |
|------|------|------|------------------------------|--------|-----------|
| FP16 (기준) | 16 | - | 5.47 | 1x | GPU |
| LLM.int8() | 8 | PTQ | ~5.48 | 2x | GPU |
| SmoothQuant | 8 (W8A8) | PTQ | ~5.55 | 2x | GPU |
| GPTQ | 4 | PTQ | 5.85 | 4x | GPU |
| AWQ | 4 | PTQ | 5.78 | 4x | GPU |
| GGUF Q4_K_M | ~4.5 | PTQ | ~5.90 | 3.6x | CPU/GPU |
| EXL2 4.0bpw | 4 | PTQ | ~5.85 | 4x | GPU |
| SqueezeLLM | 3 | PTQ | 6.18 | 5.3x | GPU |
| SpQR | ~3.1 | PTQ | ~6.10 | 5.2x | GPU |
| AQLM | 2 | PTQ | 6.81 (5.94*) | 8x | GPU |
| QuIP# | 2 | PTQ | 6.15 | 8x | GPU |
| GGUF Q2_K | ~2.6 | PTQ | ~9.08 | 6.2x | CPU/GPU |
| BitNet b1.58 | 1.58 | QAT | ~8.5 (3B 모델) | 10x | 전용HW |

\* AQLM with fine-tuning

---

## 5. 참고 논문

### 5.1 Mixed Precision 및 기초 양자화 이론

1. Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu, "Mixed Precision Training," **ICLR 2018**, https://arxiv.org/abs/1710.03740

2. Raghuraman Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper," arXiv 2018, https://arxiv.org/abs/1806.08342

3. Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling, "Data-Free Quantization Through Weight Equalization and Bias Correction," **ICCV 2019**, https://arxiv.org/abs/1906.04721

4. Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," **CVPR 2018**, https://arxiv.org/abs/1712.05877

### 5.2 LLM 전용 INT8 양자화

5. Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," **NeurIPS 2022**, https://arxiv.org/abs/2208.07339

6. Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," **ICML 2023**, https://arxiv.org/abs/2211.10438

### 5.3 OBS/OBQ 프레임워크 및 GPTQ

7. Babak Hassibi, David G. Stork, "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon," **NeurIPS 1993**

8. Elias Frantar, Sidak Pal Singh, Dan Alistarh, "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning," **NeurIPS 2022**, https://arxiv.org/abs/2208.11580

9. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," **ICLR 2023**, https://arxiv.org/abs/2210.17323

### 5.4 활성값 인식 및 중요도 기반 양자화

10. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," **MLSys 2024**, https://arxiv.org/abs/2306.00978

11. Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer, "SqueezeLLM: Dense-and-Sparse Quantization," **ICML 2024**, https://arxiv.org/abs/2306.07629

12. Tim Dettmers, Ruslan Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh, "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression," **ICLR 2024**, https://arxiv.org/abs/2306.03078

### 5.5 벡터 양자화 및 극한 양자화

13. Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa, "QuIP: 2-Bit Quantization of Large Language Models With Guarantees," **NeurIPS 2023**, https://arxiv.org/abs/2307.13304

14. Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa, "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," **ICML 2024**, https://arxiv.org/abs/2402.04396

15. Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh, "AQLM: Extreme Compression of Large Language Models via Additive Quantization," **ICML 2024**, https://arxiv.org/abs/2401.06118

16. Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, Wanxiang Che, "OneBit: Towards Extremely Low-bit Large Language Models," arXiv 2024, https://arxiv.org/abs/2402.11295

### 5.6 GGUF/GGML 및 효율적 추론

17. Georgi Gerganov, "llama.cpp: Inference of LLaMA model in pure C/C++," GitHub 2023, https://github.com/ggerganov/llama.cpp

18. turboderp, "ExLlamaV2: A fast inference library for running LLMs locally on modern consumer-class GPUs," GitHub 2023, https://github.com/turboderp/exllamav2

### 5.7 1비트 LLM 및 QAT

19. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaiguo Yin, Dongdong Zhang, Jiayu You, Furu Wei, "BitNet: Scaling 1-bit Transformers for Large Language Models," arXiv 2023, https://arxiv.org/abs/2310.11453

20. Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv 2024, https://arxiv.org/abs/2402.17764

21. Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Wayne Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra, "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models," arXiv 2024, https://arxiv.org/abs/2305.17888

22. Wei Chen, Zhiyuan Li, Tengyu Ma, "EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge," arXiv 2024, https://arxiv.org/abs/2402.10787

### 5.8 QLoRA 및 NF4

23. Tim Dettmers, Artidoro Pagnoni, Aman Srivastava, Luke Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized Language Models," **NeurIPS 2023**, https://arxiv.org/abs/2305.14314

### 5.9 양자화 이론 및 서베이

24. Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer, "A Survey of Quantization Methods for Efficient Neural Network Inference," arXiv 2021, https://arxiv.org/abs/2103.13630

25. Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He, "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers," **NeurIPS 2022**, https://arxiv.org/abs/2206.01861

26. Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, Ping Luo, "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models," **ICLR 2024**, https://arxiv.org/abs/2308.13137

27. Xiuying Wei, Yunchen Zhang, Xianyu Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu, "Outlier Suppression+: Accurate Quantization of Large Language Models by Equivalent and Optimal Shifting and Scaling," **EMNLP 2023**, https://arxiv.org/abs/2304.09145

28. Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman, "Towards Accurate Post-Training Quantization for Diffusion Models," arXiv 2023

29. Yuxin Zhang, Lirui Zhao, Mingbao Lin, Yunyun Sun, Yiwu Yao, Xingjia Pan, Ke Li, Tzung-Yu Tsai, Rongrong Ji, "Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models," arXiv 2023, https://arxiv.org/abs/2305.12356

30. Zhihang Yuan, Lin Niu, Jiawei Liu, Wenyu Liu, Xinggang Wang, Yuzhang Shang, Guangyu Sun, Qiang Wu, Jiaxiang Wu, Bingzhe Wu, "RPTQ: Reorder-Based Post-Training Quantization for Large Language Models," arXiv 2023, https://arxiv.org/abs/2304.01089

31. Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, Baris Kasikci, "Atom: Low-bit Quantization for Efficient and Accurate LLM Serving," **MLSys 2024**, https://arxiv.org/abs/2310.19102

32. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," **NeurIPS 2023**, https://arxiv.org/abs/2306.05685

33. Babak Hassibi, David G. Stork, Gregory J. Wolff, "Optimal Brain Surgeon and General Network Pruning," **IEEE International Conference on Neural Networks 1993**

34. Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort, "A White Paper on Neural Network Quantization," arXiv 2021, https://arxiv.org/abs/2106.08295

35. Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort, "Understanding and Overcoming the Challenges of Efficient Transformer Quantization," **EMNLP 2021**, https://arxiv.org/abs/2109.12948

36. Guangxuan Xiao, Ji Lin, Han Cai, Ligeng Zhu, Yujun Lin, Song Han, "Offsite-Tuning: Transfer Learning without Full Model," arXiv 2023

37. Xiuying Wei, Yunchen Zhang, Yuhang Li, Xianyu Zhang, Ruihao Gong, Jinyang Guo, Xianglong Liu, "QDrop: Randomly Dropping Quantization for Extremely Low-Bit Post-Training Quantization," **ICLR 2022**, https://arxiv.org/abs/2203.05740

38. Haotong Qin, Yifu Ding, Mingyuan Zhang, Qinghua Yan, Aishan Liu, Qingqing Dang, Ziwei Liu, Xianglong Liu, "BiBench: Benchmarking and Analyzing Network Binarization," **ICML 2023**, https://arxiv.org/abs/2301.11233
