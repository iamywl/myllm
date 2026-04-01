# Knowledge Distillation for Large Language Models (지식 증류)

## 1. 기법의 정의

Knowledge Distillation(KD)은 대규모 Teacher 모델이 학습한 지식을 소규모 Student 모델로 전이하는 모델 압축 기법이다. Teacher의 출력 확률 분포(soft labels)에는 정답 레이블(hard labels)만으로는 포착할 수 없는 클래스 간 상대적 유사도 정보, 즉 "dark knowledge"가 내재되어 있으며, Student는 이 분포를 근사함으로써 Teacher의 일반화 능력을 상속받는 것이다.

LLM 맥락에서 KD는 단순한 분류 확률 전이를 넘어, 자연어 생성 능력, 추론 체인, 지시 수행 능력 등 복합적 지식의 전이를 포괄하는 개념으로 확장되었다. 본 문서에서는 Hinton et al. (2015)의 원형적 정의에서 출발하여 LLM 시대의 다양한 KD 변형 기법들을 인과적 흐름에 따라 분석한다.

---

## 2. 기존 기법의 한계와 KD 등장 배경

### 2.1 대규모 모델 배포의 근본적 제약

2023년 이후 GPT-4 (추정 1.8T 파라미터, MoE), LLaMA-2 70B, Falcon 180B 등 대규모 언어 모델이 벤치마크를 지배하고 있으나, 이들의 실제 배포에는 구조적 한계가 존재한다.

**문제 1: 추론 비용.** 70B 파라미터 모델의 추론에는 최소 2×A100 80GB GPU가 요구되며, 이는 단일 쿼리당 비용을 수십 배 증가시키는 것이다. Llama-2 70B의 경우 token/s 처리량이 7B 대비 약 10배 낮으며, 서빙 인프라 비용은 비례 이상으로 증가한다.

**문제 2: 접근성 제약.** GPT-4, Claude 3.5 등 최고 성능 모델은 API로만 접근 가능하며, 가중치가 비공개이다. 이는 온프레미스 배포, 도메인 특화, 데이터 주권 확보를 불가능하게 하는 것이다.

**문제 3: 양자화의 한계.** INT4/INT8 양자화로 메모리 사용량을 줄일 수 있으나, 70B→7B 수준의 파라미터 축소는 양자화만으로 달성 불가능하다. 또한 극단적 양자화(2-bit 이하)는 유의미한 성능 저하를 수반하는 것이다.

**문제 4: 소형 모델의 성능 격차.** LLaMA-2 7B를 동일 데이터로 처음부터 학습하더라도 70B 모델 대비 MMLU에서 15-20pp 낮은 성능을 보이며, 이 격차는 데이터 증량만으로 해소되지 않는 것이다.

### 2.2 KD가 제공하는 해법

KD는 상기 문제들에 대한 체계적 해법을 제공한다. Teacher가 이미 학습한 데이터 표현, 의사결정 경계, 추론 패턴을 Student에게 전이함으로써, Student는 자체 용량을 초과하는 성능을 달성할 수 있다. 이는 단순히 레이블을 모방하는 것이 아니라, Teacher의 불확실성 구조(uncertainty structure)를 학습하는 것이다.

---

## 3. 주요 기법 상세

### 3.1 Classical Knowledge Distillation (Hinton et al., 2015)

#### 문제

신경망 앙상블은 개별 모델 대비 높은 일반화 성능을 보이나, 추론 시 N배의 연산이 요구된다. 단일 소형 모델로 앙상블의 성능을 근사하는 방법이 필요하였다.

#### 해법

Hinton, Vinyals, and Dean (2015)은 Temperature scaling을 활용한 soft target 학습을 제안하였다. Teacher의 logit $z_t$에 temperature $T$를 적용하여 softmax를 계산하면, 확률 분포가 평활화(smoothed)되어 비정답 클래스 간의 상대적 유사도 정보가 드러나는 것이다.

**Softened probability:**

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**KD 손실 함수:**

$$\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot \text{KL}\left(\sigma(z_t / T) \| \sigma(z_s / T)\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z_s, y)$$

여기서 $\sigma$는 softmax 함수, $z_t$와 $z_s$는 각각 Teacher와 Student의 logits, $y$는 ground truth 레이블이다. $T^2$ 계수는 temperature scaling에 의한 gradient magnitude 보정을 위해 필요한 것이다. $\alpha$는 두 손실 항의 가중치를 조절하는 하이퍼파라미터이다.

**Temperature의 효과:**
- $T = 1$: 표준 softmax, 피크 확률이 지배적인 sharp 분포
- $T \in [2, 20]$: 일반적 KD 사용 범위, dark knowledge가 효과적으로 전이됨
- $T \to \infty$: 균등 분포에 수렴, 모든 클래스 정보가 동등해져 유용한 신호가 소실됨

#### 실험 결과

MNIST에서 Teacher 앙상블(10개 모델) 대비 단일 Student가 동등한 성능을 달성하였으며, CIFAR-10에서도 유사한 결과가 관찰되었다.

#### 한계

분류(classification) 태스크를 전제로 설계되어, 자기회귀 시퀀스 생성(autoregressive sequence generation) 모델에는 직접 적용이 어려운 것이다. 각 디코딩 스텝의 조건부 확률을 독립적으로 매칭하는 token-level KD는 시퀀스 수준의 일관성을 보장하지 않는다.

---

### 3.2 DistilBERT (Sanh et al., 2019)

#### 문제

BERT-base (110M)는 당시 NLU 벤치마크에서 우수한 성능을 보였으나, 실시간 서비스 배포에는 latency가 과도하였다. 모바일 및 엣지 환경에서의 운용은 사실상 불가능하였다.

#### 해법

Sanh, Debut, Chaumond, and Wolf (2019)는 triple loss를 결합한 BERT 증류를 제안하였다.

$$\mathcal{L}_{\text{DistilBERT}} = \alpha \cdot \mathcal{L}_{\text{CE}} + \beta \cdot \mathcal{L}_{\text{MLM}} + \gamma \cdot \mathcal{L}_{\cos}}$$

- $\mathcal{L}_{\text{CE}}$: Teacher-Student 간 soft label KL divergence
- $\mathcal{L}_{\text{MLM}}$: Masked Language Model 손실 (task-specific)
- $\mathcal{L}_{\cos}$: Teacher-Student hidden state 간 cosine embedding loss

핵심 아키텍처 결정은 BERT-base의 12층에서 6층을 선택적으로 초기화(layer 0, 2, 4, 6, 8, 10 → Student layer 0-5)하는 것이었다.

#### 실험 결과

| 지표 | BERT-base | DistilBERT | 비율 |
|------|-----------|------------|------|
| 파라미터 수 | 110M | 66M | 60% |
| GLUE 평균 | 79.5 | 77.0 | 97% |
| 추론 속도 | 1× | 1.6× | - |

#### 한계

Encoder-only 아키텍처에 특화되어 있어, GPT 계열의 decoder-only 또는 T5 계열의 encoder-decoder 모델에는 직접 적용이 불가능한 것이다. 또한 사전학습 단계의 증류이므로 downstream task별 재증류가 필요하였다.

---

### 3.3 Black-box Knowledge Distillation

#### 3.3.1 문제

GPT-4 등 최고 성능 모델의 가중치와 logit이 비공개인 상황에서, API 출력 텍스트만으로 지식을 전이해야 하는 현실적 제약이 존재한다.

#### 3.3.2 Alpaca (Taori et al., 2023)

Stanford 연구진은 GPT-3.5(text-davinci-003)에 self-instruct 방식으로 52K개의 instruction-response 쌍을 생성시킨 후, LLaMA 7B를 SFT(Supervised Fine-Tuning)하였다. 총 학습 비용은 $600 미만이었으며, 이는 black-box KD의 경제적 실현 가능성을 최초로 입증한 사례이다.

**한계:** 생성된 데이터의 품질이 불균일하였으며, 복잡한 추론이나 수학 문제에서 성능 격차가 현저하였다. 또한 OpenAI의 Terms of Service에 의해 경쟁 모델 학습에 GPT 출력을 사용하는 것이 금지되어 있어 법적 리스크가 존재하였다.

#### 3.3.3 Vicuna (Chiang et al., 2023)

ShareGPT 플랫폼에서 수집한 약 70K개의 사용자-GPT-4 대화 로그를 활용하여 LLaMA 13B를 학습하였다. GPT-4 기반 자동 평가에서 ChatGPT 품질의 약 90%를 달성한 것으로 보고되었다. 다회전 대화(multi-turn conversation) 능력이 Alpaca 대비 유의미하게 향상되었다.

#### 3.3.4 Orca (Mukherjee et al., 2023)

**문제:** Alpaca, Vicuna 등 기존 black-box KD는 Teacher의 최종 답변만 모방하여, Student가 추론 과정을 학습하지 못하는 "imitation gap" 문제가 있었다. 표면적 패턴 매칭은 가능하나 복잡한 추론에서는 실패하는 것이다.

**해법:** Microsoft Research는 system message를 활용하여 GPT-4로 하여금 단계별 추론 과정(explanation traces)을 포함한 응답을 생성하게 하였다. 5M개의 복합 쿼리에 대해 GPT-4와 GPT-3.5의 설명 포함 응답을 수집하고, 13B Student 모델을 progressive learning 방식으로 학습하였다.

**결과:** BBH(Big-Bench Hard)에서 Vicuna 13B 대비 100% 이상의 성능 향상을 달성하였으며, 일부 태스크에서 ChatGPT에 근접하거나 능가하는 결과를 보였다. 이는 Teacher의 추론 과정 전이가 최종 답변 전이보다 효과적임을 실증한 것이다.

#### 3.3.5 Black-box KD의 구조적 한계

Black-box KD의 근본적 한계는 Teacher의 내부 확률 분포를 활용할 수 없다는 점이다. Teacher가 "Paris"라는 답에 0.85, "Lyon"에 0.10, "Marseille"에 0.04의 확률을 부여했을 때, black-box에서는 "Paris"라는 텍스트만 전이되며 나머지 분포 정보는 소실된다. 이는 정보 이론적으로 유의미한 손실인 것이다.

---

### 3.4 White-box Knowledge Distillation

#### 3.4.1 Token-level KD의 문제: Forward KL vs Reverse KL

자기회귀 LLM에 대한 naive한 KD는 각 디코딩 스텝 $t$에서 Teacher 분포 $P_T(\cdot | x_{<t})$와 Student 분포 $P_S(\cdot | x_{<t})$ 간의 forward KL divergence를 최소화하는 것이다.

**Forward KL (mode-covering):**

$$\text{KL}(P_T \| P_S) = \sum_x P_T(x) \log \frac{P_T(x)}{P_S(x)}$$

Forward KL은 $P_T(x) > 0$인 모든 영역에서 $P_S(x) > 0$을 요구하므로, Student가 Teacher의 전체 분포를 커버하려는 mode-covering 행동을 유발한다. 용량이 제한된 Student에게 이는 확률 질량의 과도한 분산을 초래하며, 생성 텍스트의 품질 저하로 이어지는 것이다.

**Reverse KL (mode-seeking):**

$$\text{KL}(P_S \| P_T) = \sum_x P_S(x) \log \frac{P_S(x)}{P_T(x)}$$

Reverse KL은 $P_S(x) > 0$인 영역에서만 페널티가 발생하므로, Student가 Teacher의 주요 모드(high-probability region)에 집중하는 mode-seeking 행동을 유발한다. 이는 용량 제한된 Student에게 더 적합한 특성인 것이다.

#### 3.4.2 MiniLLM (Gu et al., 2024)

**문제:** Token-level forward KL은 시퀀스 생성 모델에서 exposure bias와 결합되어 성능이 급격히 저하된다. Teacher 시퀀스에 대한 token-level matching은 Student의 실제 생성 궤적(trajectory)과 괴리가 발생하는 것이다.

**해법:** Gu, Dong, Wei, and Huang (2024)은 시퀀스 수준 reverse KL minimization을 policy gradient 방법으로 최적화하는 MiniLLM을 제안하였다.

$$\mathcal{L}_{\text{MiniLLM}} = \text{KL}(P_S \| P_T) = \mathbb{E}_{y \sim P_S} \left[ \sum_{t=1}^{|y|} \log \frac{P_S(y_t | y_{<t}, x)}{P_T(y_t | y_{<t}, x)} \right]$$

이를 직접 최적화하기 위해 REINFORCE 알고리즘의 변형을 적용하며, variance reduction을 위한 baseline과 reward clipping을 도입하였다. 추가적으로 단일 스텝 정규화(single-step regularization)를 적용하여 학습 안정성을 확보하였다.

**실험 결과:**

| Student | Teacher | 방법 | RougeL (XSum) | RougeL (CNN/DM) |
|---------|---------|------|---------------|-----------------|
| GPT-2 120M | GPT-2 1.5B | SFT | 18.4 | 26.1 |
| GPT-2 120M | GPT-2 1.5B | SeqKD | 18.9 | 26.5 |
| GPT-2 120M | GPT-2 1.5B | Forward KL | 19.1 | 26.8 |
| GPT-2 120M | GPT-2 1.5B | **MiniLLM** | **20.3** | **27.6** |

LLaMA 계열에서도 MiniLLM은 기존 KD 대비 일관된 향상을 보였다. OpenLLaMA-3B → 1.1B 증류에서 forward KL 대비 RougeL 1.2pp, GPT-4 평가에서 승률 12% 향상을 달성하였다.

#### 3.4.3 GKD: Generalized Knowledge Distillation (Agarwal et al., 2024)

**문제:** 기존 KD(MiniLLM 포함)는 off-policy 학습에 의존한다. Teacher 생성 시퀀스 또는 고정 데이터셋 상에서 Student를 학습하면, Student의 실제 생성 분포와 학습 분포 간의 불일치(distribution mismatch)가 발생하는 것이다. 이는 자기회귀 모델에서 compounding error로 축적된다.

**해법:** Agarwal, Vieillard, Stanber, Ranzato, and Sharman (2024, DeepMind)은 on-policy GKD를 제안하였다. 핵심 아이디어는 Student 자체가 생성한 시퀀스에 대해 Teacher의 token-level 확률로 피드백을 제공하는 것이다.

$$\mathcal{L}_{\text{GKD}} = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim P_S(\cdot|x)} \left[ \sum_{t=1}^{|y|} D\left(P_T(\cdot | y_{<t}, x) \| P_S(\cdot | y_{<t}, x)\right) \right]$$

여기서 $D$는 임의의 divergence measure(forward KL, reverse KL, JSD 등)이며, 핵심은 $y$가 Student로부터 샘플링된다는 점이다. 이 on-policy 방식은 Student가 실제로 생성할 시퀀스에 대한 학습을 보장하여 distribution mismatch를 해소하는 것이다.

GKD는 추가적으로 mixed sampling 전략을 도입하였다. 확률 $\lambda$로 Student 샘플, $1-\lambda$로 데이터셋 샘플을 사용하여 on-policy와 off-policy의 균형을 조절한다.

**실험 결과:** T5-XL (3B) → T5-Small (60M) 증류에서 GKD는 기존 on-policy 방법 대비 번역 태스크에서 BLEURT 1.5점 향상, 요약에서 RougeL 0.8점 향상을 달성하였다.

---

### 3.5 Distilling Step-by-Step (Hsieh et al., 2023)

#### 문제

기존 KD는 Teacher의 최종 출력만 전이하거나(black-box), token-level 확률 분포를 전이한다(white-box). 그러나 LLM의 핵심 능력 중 하나인 chain-of-thought 추론은 이 두 방식 모두로는 효과적으로 전이되지 않는 것이다. Student는 정답을 암기하되, 그 정답에 도달하는 추론 경로를 학습하지 못한다.

#### 해법

Hsieh, Li, Yeh, Nakhost, Fujii, Ratner, Krishna, Lee, and Pfister (2023, Google Research)는 multi-task learning 프레임워크를 통해 rationale과 label을 동시에 학습하는 방법을 제안하였다.

$$\mathcal{L}_{\text{Step}} = \mathcal{L}_{\text{label}}(y, \hat{y}) + \lambda \cdot \mathcal{L}_{\text{rationale}}(r, \hat{r})$$

- $\mathcal{L}_{\text{label}}$: 정답 레이블 예측 손실
- $\mathcal{L}_{\text{rationale}}$: Teacher가 생성한 추론 과정(rationale) 예측 손실
- $\lambda$: 두 태스크 간 가중치

Teacher(PaLM 540B)에게 few-shot CoT prompting으로 (질문, 추론과정, 답변) 트리플을 생성시킨 후, Student(T5 모델)를 두 가지 출력을 동시에 예측하도록 학습시킨다.

#### 실험 결과

| 데이터셋 | PaLM 540B (Few-shot) | T5-base 220M (Fine-tune) | T5-base 220M (Step-by-Step) |
|----------|----------------------|--------------------------|-----------------------------|
| e-SNLI | 87.2 | 85.3 | **88.1** |
| ANLI | 52.3 | 48.7 | **54.0** |
| CQA | 79.9 | 73.2 | **78.5** |
| SVAMP | 79.0 | 60.1 | **73.4** |

770M Student가 540B Teacher를 일부 태스크에서 능가하는 결과는, 추론 과정의 명시적 학습이 모델 규모를 보상할 수 있음을 시사하는 것이다. 또한 동일 성능 달성에 필요한 학습 데이터가 standard fine-tuning 대비 50% 이하로 감소하였다.

#### 한계

Teacher의 추론 과정이 항상 정확하지 않으며(unfaithful rationale 문제), rationale 생성 비용이 추가된다. 또한 CoT가 효과적인 태스크에 한정되며, 창의적 생성이나 대화 등에는 적용이 제한적인 것이다.

---

### 3.6 Self-Distillation: SPIN (Chen et al., 2024)

#### 문제

외부 Teacher 모델에 대한 의존은 비용, 접근성, 라이선스 측면에서 제약을 수반한다. 모델이 자체적으로 성능을 개선할 수 있는 self-improvement 메커니즘이 필요한 것이다.

#### 해법

Chen, Deng, Yuan, Ji, and Gu (2024)는 Self-Play Fine-Tuning(SPIN)을 제안하였다. SPIN의 핵심은 현재 모델이 생성한 응답(약한 응답)과 인간 작성 응답(강한 응답)을 구별하는 게임을 반복적으로 수행하는 자기 대전(self-play) 프레임워크이다.

각 반복 $t$에서:
1. 현재 모델 $\pi_t$로 프롬프트에 대한 응답을 생성
2. 생성된 응답(losing)과 인간 응답(winning)의 쌍을 구성
3. DPO와 유사한 목적함수로 $\pi_{t+1}$을 학습

$$\mathcal{L}_{\text{SPIN}}(\pi_\theta; \pi_t) = \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_{\text{real}} | x)}{\pi_t(y_{\text{real}} | x)} - \beta \log \frac{\pi_\theta(y_{\text{synth}} | x)}{\pi_t(y_{\text{synth}} | x)} \right) \right]$$

여기서 $y_{\text{real}}$은 인간 작성 응답, $y_{\text{synth}}$는 $\pi_t$가 생성한 응답이다. 이론적으로, SPIN은 모델 분포가 인간 응답 분포와 일치할 때 Nash equilibrium에 수렴하는 것이다.

#### 실험 결과

Zephyr-7B에 SPIN을 3회 반복 적용한 결과:
- Open LLM Leaderboard 평균: 58.14 → 63.16 (+5.02)
- MT-Bench: 6.78 → 7.31
- AlpacaEval 2.0: 10.0% → 14.8%

추가 외부 데이터나 Teacher 모델 없이 순수 자기 개선만으로 유의미한 성능 향상을 달성하였다.

#### 한계

자기 자신을 Teacher로 사용하므로 성능 상한이 존재한다. 인간 참조 데이터의 품질에 강하게 의존하며, 반복 횟수 증가에 따른 수확 체감(diminishing returns)이 관찰되는 것이다 (3회 이후 향상 미미).

---

### 3.7 Multi-Teacher Distillation

#### 문제

단일 Teacher는 특정 도메인이나 태스크에 편향될 수 있으며, Teacher의 오류가 Student에 그대로 전이되는 문제가 있다.

#### 해법

다수의 Teacher 모델로부터 지식을 집계(aggregate)하여 Student에게 전이하는 접근법이다. Li, Lin, Zhang, and Fu (2024)는 LLM-Blender를 통해 다수 LLM의 출력을 순위화하고 융합하는 프레임워크를 제안하였다.

**앙상블 기반 확률 집계:**

$$P_{\text{ensemble}}(y_t | y_{<t}, x) = \sum_{k=1}^{K} w_k \cdot P_{T_k}(y_t | y_{<t}, x)$$

여기서 $w_k$는 $k$번째 Teacher의 가중치이며, 태스크별로 동적으로 조정될 수 있다. 대안적으로, 각 Teacher의 출력 시퀀스 중 최고 품질을 선택하는 routing 방식도 활용된다.

Wan, Li, Yao, Chen, and Xu (2024)는 Knowledge Fusion of Large Language Models를 제안하여, 이질적 구조의 Teacher 모델(GPT-4, Claude, PaLM)로부터의 지식을 단일 Student에 융합하는 방법을 연구하였다. 각 Teacher의 토큰 확률 분포를 가중 평균하는 것이 아닌, 지식 영역별 선택적 전이를 수행하는 것이 핵심이다.

**Zephyr (Tunstall et al., 2023):**
Tunstall, Hawkins, Lambert, Narayanan, Beeching, and colleagues (2023, Hugging Face)는 GPT-4의 응답과 AI Feedback을 결합한 증류 파이프라인을 제안하였다. UltraChat 데이터셋(GPT-3.5 생성)으로 SFT 후, UltraFeedback(GPT-4 평가)으로 DPO를 적용하여 Mistral-7B 기반 Zephyr-7B를 학습하였다. MT-Bench에서 7.34를 달성하여 당시 7B 모델 중 최고 성능을 기록하였다.

#### 한계

Teacher 간 충돌하는 지식의 처리가 어려우며, Teacher 수 증가에 따른 학습 비용이 선형적으로 증가하는 것이다. 또한 각 Teacher의 최적 가중치 결정이 비자명한 문제이다.

---

### 3.8 Symbolic Knowledge Distillation

#### 문제

신경망 기반 KD는 Teacher의 지식을 암묵적(implicit) 수치 표현으로 전이한다. 이는 해석 불가능하며, 전이된 지식의 검증과 수정이 불가능한 것이다.

#### 해법

West, Bhagavatula, Hessel, Hwang, Jiang, Le Bras, Lu, Welleck, and Choi (2022)는 Symbolic Knowledge Distillation을 제안하여, Teacher LLM이 명시적 지식 구조(knowledge graph triplets, 규칙, 코드)를 생성하고, Student가 이를 학습하는 프레임워크를 구축하였다.

구체적으로, GPT-3를 Teacher로 사용하여 상식 지식 그래프(commonsense knowledge graph)를 생성한 후, 해당 지식으로 Student 모델을 학습시켰다. 생성된 symbolic knowledge는 인간이 검증, 필터링, 수정할 수 있다는 장점이 있다.

**Li, Yu, and others (2023)의 Symbolic Chain-of-Thought Distillation:**
Teacher가 생성한 CoT를 Python 코드 형태의 symbolic program으로 변환한 후 Student에게 전이하는 방법이다. 자연어 추론과 달리 symbolic program은 실행 가능하므로 정확성 검증이 자동화되는 것이다.

#### 한계

모든 지식이 symbolic 형태로 표현 가능한 것은 아니며, 자연어의 뉘앙스, 맥락 의존적 의미 등은 symbolization이 어렵다.

---

### 3.9 On-policy vs Off-policy Distillation

#### 핵심 구분

이 구분은 KD의 효과에 결정적 영향을 미치는 설계 선택이다.

**Off-policy KD:** 고정된 데이터셋(Teacher 생성 또는 기존 코퍼스) 상에서 Student를 학습한다. 표준 SFT 기반 black-box KD, token-level forward KL 등이 이에 해당한다.

$$\mathcal{L}_{\text{off-policy}} = \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{fixed}}} \left[ D(P_T(\cdot | x) \| P_S(\cdot | x)) \right]$$

**On-policy KD:** Student의 현재 정책에서 샘플링한 시퀀스에 대해 학습한다. GKD, MiniLLM(policy gradient 사용 시) 등이 이에 해당한다.

$$\mathcal{L}_{\text{on-policy}} = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim P_S(\cdot|x)} \left[ D(P_T(\cdot | y_{<t}, x) \| P_S(\cdot | y_{<t}, x)) \right]$$

**Off-policy의 문제:** 학습 시 Student가 접하는 context ($y_{<t}$)가 Teacher나 데이터셋으로부터 온 것이므로, 추론 시 Student 자체 생성의 context와 불일치가 발생한다. 이는 자기회귀 모델에서 누적 오류(compounding error)로 이어지며, 시퀀스 길이에 비례하여 심화되는 것이다.

**On-policy의 장점과 비용:** Distribution mismatch를 원천적으로 해소하나, 매 학습 스텝마다 Student로부터 시퀀스를 샘플링해야 하므로 학습 비용이 2-5배 증가한다.

Gu, Dong, Wei, and Huang (2024)과 Agarwal et al. (2024)의 실험을 종합하면, on-policy KD는 off-policy 대비 생성 품질에서 일관된 우위를 보이나, 분류나 단답형 태스크에서는 차이가 미미한 것으로 관찰되었다. 이는 시퀀스 길이가 길수록 on-policy의 이점이 증가한다는 이론적 예측과 부합하는 것이다.

**ImitKD (Lin et al., 2020):**
Lin, Tian, Li, Wang, and Li (2020)는 on-policy KD의 초기 형태를 제안하였다. Student가 생성한 시퀀스에 대해 Teacher의 auto-regressive 확률을 레이블로 사용하는 방식으로, 이후 GKD의 이론적 기반이 되었다.

---

### 3.10 DistiLLM (Ko et al., 2024)

#### 문제

기존 white-box KD 방법들은 skewed student distribution이나 off-policy 데이터 사용으로 인한 학습 불안정성 문제가 있었다. MiniLLM의 policy gradient 방식은 높은 분산(variance)으로 인해 수렴이 느리며, GKD의 on-policy 샘플링은 연산 비용이 높은 것이다.

#### 해법

Ko, Kim, and Shin (2024)은 Adaptive Off-policy Knowledge Distillation을 제안하여, off-policy 데이터를 사용하되 Student 분포와의 괴리를 importance weighting으로 보정하는 방법을 제시하였다. 추가적으로 skew KL divergence를 도입하여 forward KL과 reverse KL의 중간 지점을 탐색하였다.

$$D_{\alpha\text{-skew}}(P_S \| P_T) = \text{KL}(P_S \| \alpha P_T + (1-\alpha) P_S)$$

$\alpha = 1$이면 forward KL, $\alpha \to 0$이면 reverse KL에 근사하며, 중간 값에서 mode-covering과 mode-seeking의 균형을 달성한다.

---

### 3.11 Lion: Adversarial Distillation (Jiang et al., 2023)

#### 문제

Black-box KD에서 Teacher에게 어떤 프롬프트를 질의할지는 Student의 학습 효율에 결정적이다. 무작위 또는 고정된 프롬프트 세트는 Student의 약점을 효과적으로 보강하지 못하는 것이다.

#### 해법

Jiang, Chan, Chen, and Wang (2023)은 adversarial 프레임워크를 통해 Student의 취약 영역을 식별하고, 해당 영역에 대한 Teacher의 응답을 집중적으로 수집하는 반복적 증류 방법을 제안하였다. 각 반복에서:
1. 심판(Referee) LLM이 Student의 약점을 식별
2. 해당 약점을 대상으로 하는 hard instruction을 생성
3. Teacher가 해당 instruction에 대한 응답을 생성
4. Student가 새 데이터로 학습

이 adversarial loop는 curriculum learning과 유사한 효과를 달성하며, 무작위 샘플링 대비 데이터 효율을 2-3배 향상시키는 것으로 보고되었다.

---

### 3.12 Task-Specific Distillation 확장

#### Magister et al. (2023): Teaching Small Language Models to Reason

Magister, Mallick, Bharadwaj, Perez, Germain, and Descombes (2023)는 CoT 추론을 명시적으로 증류하는 체계적 연구를 수행하였다. PaLM 540B와 GPT-3.5의 CoT 출력으로 T5, GPT-2 계열 소형 모델을 학습한 결과, 산술 추론(GSM8K)에서 Student 성능이 최대 20pp 향상됨을 보고하였다.

#### Fu et al. (2023): Specializing Smaller Models

Fu, Peng, Sabharwal, Clark, and Khot (2023)은 범용 Teacher로부터 특정 태스크에 특화된 Student를 증류하는 방법을 연구하였다. 핵심 발견은, 태스크별로 증류하는 것이 범용 증류보다 효과적이며, 심지어 더 작은 모델로도 더 높은 태스크 성능을 달성할 수 있다는 것이다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 Phase 1: 원형적 KD (2015-2019)

Hinton et al. (2015)의 soft label 기반 KD는 분류 모델 압축의 표준 방법으로 자리잡았다. 그러나 이 방법은 고정된 출력 공간에서의 확률 매칭을 전제하므로, 가변 길이 시퀀스를 생성하는 언어 모델에는 직접 적용이 어려운 것이었다.

DistilBERT (2019)는 NLU 모델에 대한 KD의 실용적 가치를 입증하였으나, encoder-only 아키텍처에 한정되었다. **인과적 전환:** GPT-2/3의 등장으로 decoder-only 아키텍처가 주류가 되면서, 새로운 KD 패러다임이 요구되었다.

### 4.2 Phase 2: Black-box KD의 폭발적 성장 (2023)

ChatGPT/GPT-4의 등장은 두 가지 동시적 동인을 제공하였다. 첫째, 최고 성능 모델이 API 뒤에 존재하므로 white-box KD가 원천적으로 불가능하였다. 둘째, 이 API 출력의 품질이 충분히 높아 단순 SFT만으로도 의미 있는 Student 성능을 달성할 수 있었다.

Alpaca → Vicuna → Orca의 진화는 점진적 정교화의 과정이다:
- **Alpaca**: 최종 답변만 전이 → 표면적 모방에 그침
- **Vicuna**: 다회전 대화 데이터로 대화 능력 향상 → 여전히 추론 부재
- **Orca**: 추론 과정(explanation traces) 전이 → imitation gap 부분 해소

**인과적 전환:** Black-box KD의 한계(확률 분포 정보 소실, 라이선스 문제)와 LLaMA 등 오픈 가중치 모델의 등장이 white-box KD 연구를 촉발하였다.

### 4.3 Phase 3: White-box KD의 이론적 정교화 (2023-2024)

오픈 가중치 모델(LLaMA, Mistral)의 가용성은 Teacher logit에 대한 완전한 접근을 가능하게 하였다. 이 시점에서 핵심 연구 질문은 "어떤 divergence measure를, 어떤 sampling 전략으로 최적화할 것인가"로 전환되었다.

- **MiniLLM**: Forward KL → Reverse KL 전환으로 mode-seeking 유도
- **GKD**: Off-policy → On-policy 전환으로 distribution mismatch 해소
- **DistiLLM**: Skew KL로 두 극단의 균형 탐색, importance weighting으로 off-policy 효율 유지

이 세 연구는 동일한 근본 문제(자기회귀 시퀀스에 대한 분포 매칭)에 대해 상보적 해법을 제시하며, KD 설계 공간의 주요 축(divergence type × sampling policy)을 체계적으로 탐색한 것이다.

### 4.4 Phase 4: 추론 능력의 명시적 전이 (2023-2024)

Distilling Step-by-Step, Orca, Magister et al.의 연구는 공통적으로 "무엇을(what)" 전이하는가에서 "어떻게(how)" 전이하는가로 초점을 이동시켰다. Teacher의 추론 과정을 명시적으로 전이하면, Student의 용량 한계를 부분적으로 극복할 수 있다는 발견은 KD 연구의 패러다임 전환이었다. 770M 모델이 540B 모델을 능가한다는 결과는, 추론 과정의 구조적 학습이 파라미터 수에 의한 한계를 보상할 수 있음을 실증한 것이다.

### 4.5 Phase 5: Self-Distillation과 Teacher 독립성 (2024)

SPIN은 외부 Teacher 의존성을 완전히 제거하는 방향으로 진화하였다. 자기 대전을 통해 모델이 자체 약점을 점진적으로 보강하는 이 접근법은 RLHF의 self-play 개념과 KD를 결합한 것이다. 그러나 성능 상한의 존재와 수확 체감 문제는 순수 self-distillation의 구조적 한계를 보여준다.

### 4.6 현재 연구 프론티어 (2024-2025)

현재 KD 연구는 다음 방향으로 수렴하고 있다:

1. **Speculative Decoding과의 결합:** Student가 draft token을 생성하고 Teacher가 검증하는 speculative decoding에서, Student의 품질이 높을수록 acceptance rate가 증가하므로 KD는 추론 가속의 핵심 전제 기술이 되는 것이다 (Leviathan et al., 2023).

2. **Reasoning Distillation의 심화:** DeepSeek-R1 (DeepSeek, 2025)은 대규모 reasoning 모델로부터 소형 모델로의 증류를 대규모로 수행하여, 1.5B-70B 범위에서 reasoning 능력의 효과적 전이를 시연하였다.

3. **Synthetic Data Generation으로서의 KD:** Phi-1 (Gunasekar et al., 2023), Phi-2, Phi-3 계열은 GPT-4가 생성한 "textbook-quality" synthetic data로 소형 모델을 학습하는 방법을 제안하였다. 이는 KD와 데이터 증강(data augmentation)의 경계가 희미해지는 추세를 반영한 것이다.

4. **Layer-wise 및 Attention Transfer:** TinyBERT (Jiao et al., 2020)에서 시작된 중간 표현 전이가 LLM 규모로 확장되고 있으며, attention distribution의 선택적 전이가 연구되고 있다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Distilling the Knowledge in a Neural Network | Geoffrey Hinton, Oriol Vinyals, Jeff Dean | **NeurIPS 2014 Workshop** | https://arxiv.org/abs/1503.02531 |
| 2 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf | **NeurIPS 2019 Workshop** | https://arxiv.org/abs/1910.01108 |
| 3 | MiniLLM: Knowledge Distillation of Large Language Models | Yuxian Gu, Li Dong, Furu Wei, Minlie Huang | **ICLR 2024** | https://arxiv.org/abs/2306.08543 |
| 4 | GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models | Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ranzato, and others | **ICLR 2024** | https://arxiv.org/abs/2306.13649 |
| 5 | Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes | Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister | **ACL 2023 (Findings)** | https://arxiv.org/abs/2305.02301 |
| 6 | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN) | Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu | **ICML 2024** | https://arxiv.org/abs/2401.01335 |
| 7 | Lion: Adversarial Distillation of Closed-Source Large Language Model | Yuxin Jiang, Chunkit Chan, Mingyang Chen, Wei Wang | **EMNLP 2023** | https://arxiv.org/abs/2305.12870 |
| 8 | Orca: Progressive Learning from Complex Explanation Traces of GPT-4 | Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, Ahmed Awadallah | **arXiv 2023** | https://arxiv.org/abs/2306.02707 |
| 9 | Zephyr: Direct Distillation of LM Alignment | Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Alexander Rush, Thomas Wolf | **arXiv 2023** | https://arxiv.org/abs/2310.16944 |
| 10 | A Survey on Knowledge Distillation of Large Language Models | Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, Tianyi Zhou | **arXiv 2024** | https://arxiv.org/abs/2402.13116 |
| 11 | Alpaca: A Strong, Replicable Instruction-Following Model | Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori Hashimoto | **Stanford CRFM 2023** | https://crfm.stanford.edu/2023/03/13/alpaca.html |
| 12 | Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality | Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph Gonzalez, Ion Stoica, Eric Xing | **LMSYS 2023** | https://lmsys.org/blog/2023-03-30-vicuna/ |
| 13 | Symbolic Knowledge Distillation: from General Language Models to Commonsense Models | Peter West, Chandra Bhagavatula, Jack Hessel, Jena Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi | **NAACL 2022** | https://arxiv.org/abs/2110.07178 |
| 14 | Teaching Small Language Models to Reason | Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, Aliaksei Severyn | **ACL 2023** | https://arxiv.org/abs/2212.08410 |
| 15 | Specializing Smaller Language Models towards Multi-Step Reasoning | Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot | **ICML 2023** | https://arxiv.org/abs/2301.12726 |
| 16 | ImitKD: Transferring Knowledge via Imitation Learning | Xinyu Lin, Shao-Lun Huang, Youzhi Liang, Tianyu Li, Hao Wang, Yanyong Zhang | **arXiv 2020** | https://arxiv.org/abs/2011.13549 |
| 17 | TinyBERT: Distilling BERT for Natural Language Understanding | Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu | **EMNLP 2020 (Findings)** | https://arxiv.org/abs/1909.10351 |
| 18 | DistiLLM: Towards Streamlined Distillation for Large Language Models | Jongwoo Ko, Sungnyun Kim, Sangwan Lee, Se-Young Yun | **ICML 2024** | https://arxiv.org/abs/2402.03898 |
| 19 | Knowledge Fusion of Large Language Models | Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, Shuming Shi | **ICLR 2024** | https://arxiv.org/abs/2401.10491 |
| 20 | LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion | Dongfu Jiang, Xiang Ren, Bill Yuchen Lin | **ACL 2023** | https://arxiv.org/abs/2306.02561 |
| 21 | Textbooks Are All You Need (Phi-1) | Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, and others | **arXiv 2023** | https://arxiv.org/abs/2306.11644 |
| 22 | Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2Seq Generation | Yaniv Leviathan, Matan Kalman, Yossi Matias | **ICML 2023** | https://arxiv.org/abs/2211.17192 |
| 23 | Patient Knowledge Distillation for BERT Model Compression | Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu | **EMNLP 2019** | https://arxiv.org/abs/1908.09355 |
| 24 | Born Again Neural Networks | Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar | **ICML 2018** | https://arxiv.org/abs/1805.04770 |
| 25 | Sequence-Level Knowledge Distillation | Yoon Kim, Alexander Rush | **EMNLP 2016** | https://arxiv.org/abs/1606.07947 |
| 26 | FitNets: Hints for Thin Deep Nets | Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio | **ICLR 2015** | https://arxiv.org/abs/1412.6550 |
| 27 | Orca 2: Teaching Small Language Models How to Reason | Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah | **arXiv 2023** | https://arxiv.org/abs/2311.11045 |
| 28 | Distill or Not to Distill: A Comparative Study of Knowledge Distillation Methods for LLMs | Tianjun Zhang, Shishir Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph Gonzalez | **arXiv 2024** | https://arxiv.org/abs/2402.13116 |
| 29 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies | Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, and others | **arXiv 2024** | https://arxiv.org/abs/2404.06395 |
| 30 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | **arXiv 2025** | https://arxiv.org/abs/2501.12948 |
| 31 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | Rishabh Agarwal, Nino Vieillard, Piotr Stanczyk, Sabela Ranzato, Olivier Bachem | **ICLR 2024** | https://arxiv.org/abs/2306.13649 |
| 32 | The Wisdom of Hindsight Makes Language Models Better Instruction Followers | Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, Joseph Gonzalez | **ICML 2023** | https://arxiv.org/abs/2302.05206 |
| 33 | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, and others | **arXiv 2024** | https://arxiv.org/abs/2404.14219 |
| 34 | WizardLM: Empowering Large Language Models to Follow Complex Instructions | Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Daxin Jiang | **ICLR 2024** | https://arxiv.org/abs/2304.12244 |
| 35 | Contrastive Decoding: Open-ended Text Generation as Optimization | Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis | **ACL 2023** | https://arxiv.org/abs/2210.15097 |

---

*본 문서는 Knowledge Distillation의 이론적 기반에서 LLM 시대의 실용적 변형까지를 포괄하며, 각 기법이 해결한 문제와 새로이 야기한 한계를 인과적으로 추적하였다. KD는 모델 압축의 수단을 넘어, synthetic data generation, reasoning transfer, self-improvement의 통합 프레임워크로 진화하고 있는 것이다.*
