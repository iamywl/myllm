# RLHF & Preference Optimization (강화학습 기반 인간 선호도 정렬)

## 1. 기법의 정의

### 1.1 RLHF의 형식적 정의

RLHF(Reinforcement Learning from Human Feedback)는 인간이 제공한 선호도 비교 데이터를 학습 신호로 활용하여, 사전학습된 언어 모델의 출력 분포를 인간의 의도에 정렬(align)시키는 강화학습 기반 최적화 프레임워크이다. 형식적으로 정의하면, 프롬프트 $x \sim \mathcal{D}$에 대해 정책 $\pi_\theta$가 생성하는 응답 $y$의 기대 보상을 극대화하되, 참조 정책 $\pi_{\text{ref}}$로부터의 KL 발산을 제약하는 최적화 문제이다:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}}\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]$$

여기서 $r_\phi(x, y)$는 인간 선호도 데이터로 학습된 보상 모델이며, $\beta > 0$는 KL 페널티 계수이다.

Preference Optimization은 RLHF를 포함하는 상위 개념으로, 인간 또는 AI가 제공한 선호도 신호를 활용하여 언어 모델을 최적화하는 모든 기법을 포괄하는 용어이다. DPO, KTO, ORPO 등 보상 모델을 명시적으로 학습하지 않는 기법들도 이 범주에 포함된다.

### 1.2 핵심 구성 요소

RLHF 및 선호도 최적화 기법의 구성 요소는 다음과 같이 분류된다:

1. **선호도 데이터**: 동일 프롬프트 $x$에 대한 두 응답 $(y_w, y_l)$의 순서쌍. $y_w$는 선호(chosen) 응답, $y_l$은 비선호(rejected) 응답이다.
2. **보상 모델 $r_\phi$**: Bradley-Terry 모델 기반으로 인간 선호도를 스칼라 보상으로 변환하는 함수이다.
3. **정책 모델 $\pi_\theta$**: 최적화 대상인 언어 모델이다.
4. **참조 모델 $\pi_{\text{ref}}$**: 일반적으로 SFT 단계 이후의 모델로, KL 제약의 기준점이다.
5. **KL 발산 제약**: 정책이 참조 모델에서 과도하게 이탈하는 것을 방지하는 정규화 항이다.

---

## 2. 기존 기법의 한계와 RLHF 등장 배경

### 2.1 사전학습 및 지도 미세조정의 근본적 한계

**문제 1: 목적함수의 불일치 (Objective Mismatch)**

사전학습의 목적함수는 다음 토큰 예측의 교차 엔트로피 최소화이다:

$$\mathcal{L}_{\text{PT}} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

이 목적함수는 인터넷 텍스트의 통계적 분포를 학습할 뿐, "인간에게 도움이 되는 응답"이라는 가치 판단을 포함하지 않는다. 따라서 사전학습 모델은 유해한 콘텐츠, 허위 정보, 편향된 텍스트를 구분 없이 생성하는 문제가 발생한다 (Bender et al., 2021).

**문제 2: 지도 미세조정(SFT)의 한계**

SFT는 고품질 시연(demonstration) 데이터에 대한 행동 복제(behavior cloning)에 해당한다. 그러나 행동 복제는 다음과 같은 근본적 한계를 가진다:

- **분포 이동(Distribution Shift)**: 학습 데이터에 없는 프롬프트에 대해 일반화가 취약하다.
- **최대 우도 추정의 한계**: 시연자의 평균적 행동을 학습할 뿐, 최적의 행동을 학습하지 못한다. 특히 시연 데이터에 품질 편차가 존재할 경우, SFT 모델은 고품질 응답과 저품질 응답을 구분하지 못하다.
- **자동 평가 메트릭의 부재**: BLEU, ROUGE 등 기존 자연어 생성 메트릭은 개방형 대화의 품질을 측정하는 데 부적합하다 (Liu et al., 2016).

**문제 3: 인간 선호도의 비정형성**

"좋은 응답"의 정의는 다차원적이며(정확성, 유용성, 안전성, 간결성 등), 단일 스칼라 보상 함수로 사전에 정의하는 것이 불가능하다. 이는 보상 함수를 인간 피드백으로부터 학습해야 한다는 근거를 제공한다.

### 2.2 RLHF의 이론적 기원

RLHF의 핵심 아이디어는 Christiano et al. (2017)이 **NeurIPS 2017**에서 제안한 "Deep Reinforcement Learning from Human Preferences"에서 기원하였다. 해당 연구는 Atari 게임 및 MuJoCo 시뮬레이션 환경에서 인간이 두 궤적(trajectory) 중 더 나은 것을 선택하는 비교 피드백만으로 보상 함수를 학습하고, 이를 기반으로 강화학습 에이전트를 훈련할 수 있음을 실증하였다.

그러나 이 초기 연구는 소규모 제어 태스크에 한정되었으며, 대규모 언어 모델에의 적용은 Stiennon et al. (2020)이 **NeurIPS 2020**에서 텍스트 요약 태스크에 RLHF를 적용한 "Learning to Summarize from Human Feedback"에서 처음 시도되었다. 이 연구는 GPT-3 기반 모델에 인간 선호도 비교 데이터를 활용한 PPO 학습이 기존 지도학습 대비 요약 품질을 유의미하게 향상시킨다는 결과를 보고하였다.

### 2.3 Bradley-Terry 모델과 보상 모델 학습

선호도 비교 데이터를 보상 함수로 변환하는 핵심 도구는 Bradley-Terry 모델(Bradley and Terry, 1952)이다. 이 모델은 두 응답 $(y_1, y_2)$에 대한 인간의 선호 확률을 보상 함수의 차이로 모델링한다:

$$P(y_1 \succ y_2 | x) = \sigma\left(r_\phi(x, y_1) - r_\phi(x, y_2)\right)$$

여기서 $\sigma$는 시그모이드 함수이다. 보상 모델의 학습 목적함수는 인간 선호도 데이터셋 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$에 대한 음의 로그 우도 최소화이다:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

**보상 모델 학습의 실질적 고려사항:**

- 보상 모델은 일반적으로 정책 모델과 동일 아키텍처에서 최종 토큰의 스칼라 출력을 보상 값으로 사용한다 (Ouyang et al., 2022).
- 학습 데이터의 라벨러 간 합의율(inter-annotator agreement)은 약 65-75% 수준이며, 이는 보상 모델의 성능 상한을 결정하는 핵심 요인이다.
- 보상 모델의 과적합을 방지하기 위해 별도의 검증 세트에서 정확도를 모니터링해야 하며, Ouyang et al. (2022)는 6B 보상 모델이 175B 정책 모델에도 효과적임을 보고하였다.

---

## 3. 주요 기법 상세

### 3.1 InstructGPT와 RLHF 3단계 파이프라인

Ouyang et al. (2022)이 **NeurIPS 2022**에서 발표한 InstructGPT는 RLHF를 대규모 언어 모델(GPT-3, 175B)에 최초로 적용한 연구이며, ChatGPT의 기반 기술이다.

**[1단계] Supervised Fine-Tuning (SFT)**

고품질 시연 데이터(약 13K 샘플)를 수집하여 GPT-3을 지도학습으로 미세조정한다. 시연 데이터는 40명의 전문 라벨러가 작성하였으며, OpenAI API에 제출된 실제 프롬프트를 기반으로 구성되었다.

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})$$

**[2단계] Reward Model (RM) 학습**

동일 프롬프트에 대해 SFT 모델이 생성한 복수 응답에 대한 인간 순위 라벨링(K=4~9개 응답을 순위로 정렬)을 수행한다. 순위 데이터에서 $\binom{K}{2}$개의 쌍별 비교를 추출하여 Bradley-Terry 모델 기반 보상 모델을 학습한다. InstructGPT는 약 33K 프롬프트에 대한 비교 데이터를 사용하였다.

**[3단계] PPO (Proximal Policy Optimization) 학습**

Schulman et al. (2017)이 제안한 PPO 알고리즘을 적용하여 보상 모델의 보상을 극대화하도록 정책을 최적화한다. PPO의 클리핑된 목적함수는 다음과 같다:

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]$$

여기서 $\hat{A}_t$는 GAE(Generalized Advantage Estimation)로 추정된 이점 함수이며, $\epsilon$은 클리핑 범위(일반적으로 0.2)이다. 언어 모델 RLHF에서의 보상은 다음과 같이 정의된다:

$$R(x, y) = r_\phi(x, y) - \beta \cdot D_{\text{KL}}\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]$$

KL 페널티 항은 토큰 레벨에서 계산되며, 각 토큰 $y_t$에 대해 $\beta \cdot \left[\log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t})\right]$를 보상에서 차감하는 방식으로 구현된다.

**InstructGPT의 핵심 결과:**
- 1.3B InstructGPT가 175B GPT-3보다 인간 평가에서 우세하다는 결과를 보고하였다.
- 라벨러 선호도 승률이 GPT-3 대비 85% 이상이었다.
- 그러나 TruthfulQA 등 사실성 벤치마크에서의 개선은 제한적이었다.

### 3.2 PPO 기반 RLHF의 문제점 분석

**문제 1: 보상 해킹 (Reward Hacking)**

보상 모델 $r_\phi$는 인간 선호도의 불완전한 프록시(proxy)이다. 정책이 보상 모델의 점수를 극대화하는 방향으로 최적화되면, 보상 모델의 약점을 악용하여 실제 품질은 낮지만 보상 점수가 높은 응답을 생성하게 된다. Gao et al. (2023)은 **ICML 2023**에서 보상 모델의 스케일과 보상 과최적화(reward overoptimization) 현상의 관계를 체계적으로 분석하였다. 해당 연구에 따르면, 보상 모델 점수(proxy reward)는 KL 발산이 증가함에 따라 지속적으로 상승하지만, 실제 품질(gold reward)은 특정 KL 임계값을 넘으면 하락하는 "goodharting" 현상이 관찰된다.

$$\text{Proxy reward} \uparrow \quad \text{while} \quad \text{Gold reward} \downarrow \quad \text{when} \quad D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] > \tau_{\text{critical}}$$

**문제 2: KL 발산 페널티의 딜레마**

$\beta$ 값이 너무 크면 정책이 참조 모델에서 벗어나지 못하여 학습이 진행되지 않고, 너무 작으면 보상 해킹이 발생한다. 적절한 $\beta$ 값은 태스크, 모델 크기, 보상 모델 품질에 따라 달라지며, 실질적으로 하이퍼파라미터 탐색 비용이 크다. Adaptive KL penalty(Ziegler et al., 2019)가 제안되었으나, 근본적 해결책은 아니다.

**문제 3: 학습 불안정성**

PPO는 4개의 모델을 동시에 메모리에 유지해야 한다: (1) 정책 모델 $\pi_\theta$, (2) 참조 모델 $\pi_{\text{ref}}$, (3) 보상 모델 $r_\phi$, (4) 가치 함수(critic) $V_\psi$. 이는 70B 모델 기준으로 수백 GB의 GPU 메모리를 요구하며, 학습 과정에서 4개 모델 간의 상호작용으로 인한 불안정성이 빈번하게 발생한다 (Zheng et al., 2023).

**문제 4: 데이터 효율성**

각 PPO 업데이트마다 정책 모델이 새로운 응답을 생성해야 하므로(on-policy), 추론 비용이 학습 비용의 상당 부분을 차지한다. 이는 대규모 모델에서의 RLHF 적용을 비용 측면에서 제약하는 핵심 요인이다.

### 3.3 DPO (Direct Preference Optimization)

**문제:** RLHF의 3단계 파이프라인이 복잡하고, PPO 학습이 불안정하며, 보상 모델을 별도로 학습·유지해야 하는 비용이 과도하다.

**해결:** Rafailov et al. (2023)은 **NeurIPS 2023**에서 보상 함수와 최적 정책 사이의 해석적(closed-form) 관계를 유도하여, 보상 모델 없이 선호도 데이터로 직접 정책을 최적화하는 DPO를 제안하였다.

핵심 유도 과정은 다음과 같다. KL 제약이 있는 보상 극대화 문제의 최적해는:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

이를 보상 $r$에 대해 재정리하면:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

이 관계를 Bradley-Terry 모델에 대입하면, 분배 함수 $Z(x)$가 소거되어 다음의 DPO 목적함수를 얻는다:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

**DPO의 기울기 분석:**

DPO 기울기를 전개하면, 암묵적 보상(implicit reward) $\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$에 의해 잘못 순위가 매겨진 쌍에 대해 더 큰 기울기가 부여된다는 것을 확인할 수 있다. 이는 DPO가 자동으로 어려운 사례에 집중하는 효과를 가짐을 의미한다.

**DPO에서 발생하는 새로운 문제:**

1. **오프라인 학습 한계**: DPO는 학습 데이터가 고정되어 있어 정책이 업데이트됨에 따라 학습 데이터의 분포와 현재 정책 분포 사이의 괴리(distribution shift)가 증가한다.
2. **참조 모델 의존성**: $\pi_{\text{ref}}$를 메모리에 유지해야 하므로 GPU 메모리 비용이 증가한다.
3. **과적합 경향**: 소규모 선호도 데이터셋에서 DPO는 선호 응답의 우도를 극대화하기보다 비선호 응답의 우도를 극단적으로 감소시키는 방향으로 학습하는 경향이 있다 (Azar et al., 2024).

### 3.4 IPO (Identity Preference Optimization)

**문제:** DPO는 Bradley-Terry 모델의 가정에 의존하며, 선호도 데이터에 과적합되는 경향이 있다. 특히 선호 쌍의 품질 차이가 미미한 경우에도 동일한 학습 신호를 부여한다.

**해결:** Azar et al. (2024)은 **AISTATS 2024**에서 Bradley-Terry 모델 가정을 제거하고, 직접적으로 쌍별 선호 확률에 대한 정규화된 회귀 목적함수를 제안하였다:

$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

IPO의 회귀 형태 목적함수는 DPO의 로지스틱 손실과 달리 선호 로그비의 크기를 명시적으로 제한하므로, 과적합에 대한 내재적 정규화 효과를 제공한다. 그러나 IPO는 DPO 대비 성능 향상이 일관적이지 않으며, $\beta$ 값에 대한 민감도가 높다는 한계가 보고되었다.

### 3.5 KTO (Kahneman-Tversky Optimization)

**문제:** DPO 및 그 변형들은 모두 쌍별(pairwise) 비교 데이터를 필요로 한다. 그러나 실제 환경에서 동일 프롬프트에 대한 선호/비선호 응답 쌍을 구축하는 것은 비용이 높으며, 대부분의 피드백 데이터는 단순한 "좋음/나쁨(thumbs up/down)" 형태이다.

**해결:** Ethayarajh et al. (2024)은 **ICML 2024**에서 행동경제학의 전망 이론(Kahneman & Tversky, 1979)에 기반한 KTO를 제안하였다. KTO는 쌍별 비교가 아닌 단일 응답에 대한 이진 피드백(desirable/undesirable)만으로 학습이 가능하다.

KTO의 핵심 아이디어는 인간이 동일한 크기의 이득과 손실을 비대칭적으로 인식한다는 전망 이론의 손실 회피(loss aversion) 원리를 활용하는 것이다. 목적함수는 다음과 같다:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{x, y \sim \mathcal{D}} \left[\lambda_y - v(x, y; \theta)\right]$$

여기서 가치 함수 $v$는 다음과 같이 정의된다:

$$v(x, y; \theta) = \begin{cases} \lambda_D \cdot \sigma\left(\beta \left(r_\theta(x, y) - z_{\text{ref}}\right)\right) & \text{if } y \text{ is desirable} \\ \lambda_U \cdot \sigma\left(\beta \left(z_{\text{ref}} - r_\theta(x, y)\right)\right) & \text{if } y \text{ is undesirable} \end{cases}$$

$r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$는 암묵적 보상이고, $z_{\text{ref}} = \mathbb{E}_{x', y' \sim \mathcal{D}}[r_\theta(x', y')]$는 참조점(reference point)이다. $\lambda_D$와 $\lambda_U$는 각각 desirable/undesirable 가중치이며, $\lambda_U > \lambda_D$로 설정하여 손실 회피 효과를 모사한다.

**새로운 문제:** KTO는 참조점 $z_{\text{ref}}$ 추정에 배치 전체의 통계량을 사용하므로, 배치 크기에 민감하며, 분포가 불균형한 데이터에서 학습이 불안정해질 수 있다.

### 3.6 ORPO (Odds Ratio Preference Optimization)

**문제:** DPO를 포함한 기존 기법들은 SFT 단계를 선행 조건으로 요구하므로, 전체 학습 파이프라인이 최소 2단계(SFT → DPO)로 구성된다. 이는 학습 비용 증가 및 파이프라인 관리 복잡성을 야기한다.

**해결:** Hong et al. (2024)은 **EMNLP 2024**에서 SFT와 선호도 최적화를 단일 단계로 통합하는 ORPO를 제안하였다. ORPO의 핵심은 참조 모델 없이 승산비(odds ratio)를 활용하여 선호 응답과 비선호 응답을 차별화하는 것이다:

$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \lambda \cdot \mathcal{L}_{\text{OR}}(\theta)$$

여기서 승산비 손실은:

$$\mathcal{L}_{\text{OR}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)\right]$$

$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$이며, $P_\theta(y|x)$는 토큰 단위 평균 우도이다.

ORPO의 장점은 참조 모델이 불필요하므로 메모리 비용이 절감되고, 단일 단계로 학습이 완료되어 파이프라인이 단순화된다는 것이다.

**새로운 문제:** ORPO는 대규모 모델(70B 이상)에서의 검증이 제한적이며, SFT 목적함수와 선호도 목적함수 간의 가중치 $\lambda$ 조절이 성능에 민감한 영향을 미친다.

### 3.7 SimPO (Simple Preference Optimization)

**문제:** DPO는 참조 모델 $\pi_{\text{ref}}$를 메모리에 유지해야 하며, 암묵적 보상 $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$가 실제 생성 품질과의 상관이 완벽하지 않다.

**해결:** Meng et al. (2024)은 참조 모델을 제거하고, 응답의 평균 로그 확률을 보상으로 직접 사용하는 SimPO를 제안하였다:

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma\right)\right]$$

여기서 $|y|$는 응답의 토큰 수이며, 길이 정규화를 통해 긴 응답에 대한 편향을 제거한다. $\gamma$는 마진 항으로, 선호 응답과 비선호 응답 사이의 최소 보상 차이를 보장한다.

SimPO는 참조 모델 없이 DPO와 동등하거나 우수한 성능을 달성하며, 구현의 단순성이 장점이다.

### 3.8 GRPO (Group Relative Policy Optimization)

**문제:** PPO는 가치 함수(critic) $V_\psi$를 별도로 학습해야 하며, 이는 정책 모델과 동일 규모의 추가 모델을 요구한다. 이 문제는 수십억~수천억 파라미터 모델에서 실질적인 병목이다.

**해결:** DeepSeek AI (2024, 2025)는 DeepSeek-Math 및 DeepSeek-R1에서 GRPO를 제안하였다. GRPO의 핵심은 크리틱 모델을 제거하고, 동일 프롬프트에 대한 그룹 내 상대적 보상을 이점 추정치로 사용하는 것이다.

프롬프트 $x$에 대해 $G$개의 응답 $\{y_1, y_2, \ldots, y_G\}$를 현재 정책 $\pi_{\theta_{\text{old}}}$에서 샘플링한 후, 각 응답의 정규화된 이점을 계산한다:

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}$$

GRPO의 목적함수는 PPO 스타일의 클리핑을 적용한다:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right) - \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

GRPO의 보상 함수는 규칙 기반(rule-based)으로 정의될 수 있다. DeepSeek-R1에서는 수학 문제의 정답 여부, 코드의 테스트 통과 여부 등 검증 가능한 보상을 사용하였으며, 이는 보상 모델의 불완전성에 의존하지 않는 장점이 있다.

**GRPO의 핵심적 기여:**
- 크리틱 모델 제거로 메모리 비용 약 50% 절감이다.
- DeepSeek-R1은 순수 RL(GRPO)만으로도 chain-of-thought 추론 능력이 자발적으로 출현(emergent)함을 보고하였다.
- 그룹 크기 $G$의 선택이 분산-편향 트레이드오프에 영향을 미친다.

**새로운 문제:** GRPO는 검증 가능한 보상이 존재하는 도메인(수학, 코드)에서 가장 효과적이며, 개방형 대화와 같이 보상을 명확히 정의하기 어려운 도메인에서의 효과는 제한적일 수 있다.

### 3.9 Online DPO

**문제:** DPO는 오프라인(off-policy) 알고리즘으로, 고정된 데이터셋에서만 학습한다. 정책이 업데이트됨에 따라 학습 데이터의 분포와 현재 정책의 분포 사이의 괴리가 커지며, 이는 성능 저하의 원인이 된다.

**해결:** Guo et al. (2024)은 학습 과정에서 현재 정책으로부터 새로운 응답을 생성하고, 보상 모델 또는 AI 판별자를 통해 선호도 라벨을 부여하여 DPO 학습 데이터를 지속적으로 갱신하는 Online DPO를 제안하였다. Xu et al. (2024)은 **NeurIPS 2024**에서 Online Iterative RLHF가 오프라인 DPO 대비 일관적으로 우수한 성능을 달성함을 실증적으로 보고하였다.

Online DPO의 핵심 절차는 다음과 같다:
1. 현재 정책 $\pi_\theta$에서 프롬프트 $x$에 대해 복수 응답 생성
2. 보상 모델 또는 LLM-as-a-Judge를 통해 선호도 쌍 구성
3. DPO 목적함수로 정책 업데이트
4. 1-3을 반복

이 접근은 DPO의 단순성과 PPO의 on-policy 학습의 장점을 결합하지만, 반복적 생성 및 판별 과정의 연산 비용이 추가되는 단점이 있다.

### 3.10 RLAIF (RL from AI Feedback)

**문제:** 인간 선호도 라벨링은 비용이 높고(시간당 $15-50), 라벨러 간 합의율이 낮으며, 전문 지식이 요구되는 도메인(의료, 법률 등)에서는 확보가 더욱 어렵다.

**해결:** Lee et al. (2023)은 **ICML 2024**에서 인간 라벨러 대신 LLM을 선호도 판별자로 활용하는 RLAIF를 제안하였다. 해당 연구는 PaLM 2 기반 실험에서 RLAIF가 인간 피드백 기반 RLHF와 동등한 성능을 달성함을 보고하였다.

RLAIF의 구현 방식은 두 가지로 분류된다:

1. **Distilled RLAIF**: AI가 생성한 선호도 라벨로 보상 모델을 학습한 후 PPO/DPO를 적용한다.
2. **Direct RLAIF**: LLM의 선호 확률을 직접 보상 신호로 사용한다. 구체적으로, 프롬프트 $x$와 두 응답 $y_1, y_2$에 대해 LLM이 출력하는 "1" 또는 "2" 토큰의 로그 확률을 보상으로 변환한다.

Bai et al. (2022)이 **arXiv**에서 발표한 Constitutional AI(Anthropic)는 RLAIF의 확장으로, 사전에 정의된 "헌법(constitution)" 원칙에 따라 AI가 자체적으로 응답을 비평·수정하고, 수정된 응답에 대한 선호도 라벨을 생성하는 자기 개선(self-improvement) 프레임워크이다.

**새로운 문제:** RLAIF는 판별자 LLM의 편향(position bias, verbosity bias 등)을 선호도 데이터에 전파할 위험이 있으며, 이는 "모델 붕괴(model collapse)"와 유사한 자기 강화 편향을 야기할 수 있다 (Shumailov et al., 2024).

### 3.11 Self-Play 및 SPIN

**문제:** 고품질 선호도 데이터의 지속적 확보가 어렵고, 정적 데이터셋에 기반한 학습은 모델의 잠재적 성능 상한을 제약한다.

**해결:** Chen et al. (2024)은 **ICML 2024**에서 SPIN(Self-Play Fine-Tuning)을 제안하였다. SPIN에서는 이전 반복(iteration)의 정책이 생성한 응답을 비선호 응답으로, 인간 작성 응답을 선호 응답으로 사용하여 DPO 학습을 반복한다. 이 자기 대전(self-play) 메커니즘은 별도의 보상 모델이나 인간 라벨링 없이도 반복적 개선을 가능하게 한다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 제1세대: RLHF의 확립 (2017-2022)

```
인간 선호도 → 보상 모델 → RL 학습의 3단계 파이프라인 확립

Christiano et al. (2017): 인간 선호도 비교로 보상 함수 학습 가능성 실증
    ↓ [소규모 제어 태스크 → 대규모 언어 모델 확장 필요]
Ziegler et al. (2019): GPT-2 기반 텍스트 생성에 RLHF 적용
    ↓ [텍스트 생성 → 요약/대화 등 실용적 태스크 확장 필요]
Stiennon et al. (2020): 텍스트 요약에 RLHF 적용, 인간 평가 우수성 실증
    ↓ [요약 → 범용 대화 모델 확장 필요]
Ouyang et al. (2022): InstructGPT, 175B GPT-3에 RLHF 대규모 적용
    → ChatGPT의 기반 기술로 상용화 성공
```

이 시기의 핵심 기여는 "인간 선호도를 직접 학습 신호로 사용"하는 패러다임의 확립이다. 그러나 PPO 기반 파이프라인의 복잡성, 불안정성, 비용 문제가 후속 연구의 주요 동기를 제공하였다.

### 4.2 제2세대: 보상 모델 제거 (2023)

```
PPO의 복잡성 → 보상 모델 없는 직접 최적화 기법 등장

DPO (Rafailov et al., 2023): 보상 모델-정책 간 해석적 관계 유도
    → 보상 모델 학습 불필요, 분류 손실로 단순화
    ↓ [오프라인 학습 한계, 과적합 문제 발견]
IPO (Azar et al., 2024): Bradley-Terry 가정 제거, 정규화된 회귀 목적함수
    → DPO의 과적합 완화
    ↓ [여전히 쌍별 비교 데이터 필요]
SLiC-HF (Zhao et al., 2023): 대조 학습 기반 랭킹 손실로 선호도 학습
    → DPO와 독립적으로 유사한 접근 제시
```

이 시기의 핵심 통찰은 "보상 모델은 선호도 최적화의 필요충분조건이 아니다"라는 것이다. 보상 모델의 제거는 파이프라인 단순화, 학습 안정성 향상, 하이퍼파라미터 감소 등의 실질적 이점을 제공하였다.

### 4.3 제3세대: 데이터 효율성 및 단일 단계 통합 (2024)

```
쌍별 비교 데이터 구축 비용 → 단일 응답 피드백 활용
다단계 파이프라인 → 단일 단계 통합

KTO (Ethayarajh et al., 2024): 쌍별 비교 불필요, 이진 피드백만으로 학습
    → 데이터 구축 비용 대폭 절감
ORPO (Hong et al., 2024): SFT + 선호도 최적화를 단일 손실 함수로 통합
    → 학습 단계 축소, 참조 모델 불필요
SimPO (Meng et al., 2024): 참조 모델 제거, 길이 정규화 보상
    → DPO 대비 메모리 절감, 구현 단순화
```

### 4.4 제4세대: 추론 강화 RL과 온라인 학습 (2024-2025)

```
오프라인 학습의 한계 → 온라인/반복적 학습
보상 모델의 불완전성 → 검증 가능한 보상 활용

Online DPO (Guo et al., 2024; Xu et al., 2024): 현재 정책에서 생성 + DPO 반복
    → 분포 이동 문제 완화
GRPO (DeepSeek, 2024-2025): 크리틱 모델 제거, 그룹 상대적 이점 추정
    → 수학/코드 도메인에서 추론 능력 강화
    → DeepSeek-R1: 순수 RL만으로 chain-of-thought 출현
RLAIF (Lee et al., 2023): AI 피드백으로 인간 라벨링 대체
    → 선호도 데이터 확보 비용 절감
    → Constitutional AI (Bai et al., 2022): 원칙 기반 자기 개선
```

이 시기의 핵심 발견은 두 가지이다: (1) 검증 가능한 도메인에서 RL 기반 학습이 복잡한 추론 능력의 자발적 출현을 유도할 수 있다는 것, (2) 온라인 학습이 오프라인 학습의 근본적 한계를 극복하는 데 효과적이라는 것이다.

### 4.5 기법 간 비교 종합

| 기법 | 보상 모델 | 참조 모델 | 크리틱 모델 | 데이터 형태 | 온라인/오프라인 | 메모리 비용 |
|------|:---------:|:---------:|:-----------:|:-----------:|:--------------:|:-----------:|
| **RLHF (PPO)** | 필요 | 필요 | 필요 | 쌍별 비교 | 온라인 | 매우 높음 |
| **DPO** | 불필요 | 필요 | 불필요 | 쌍별 비교 | 오프라인 | 높음 |
| **IPO** | 불필요 | 필요 | 불필요 | 쌍별 비교 | 오프라인 | 높음 |
| **KTO** | 불필요 | 필요 | 불필요 | 이진 피드백 | 오프라인 | 높음 |
| **ORPO** | 불필요 | 불필요 | 불필요 | 쌍별 비교 | 오프라인 | 낮음 |
| **SimPO** | 불필요 | 불필요 | 불필요 | 쌍별 비교 | 오프라인 | 낮음 |
| **GRPO** | 외부/규칙 | 필요 | 불필요 | 그룹 보상 | 온라인 | 중간 |
| **Online DPO** | 외부/AI | 필요 | 불필요 | 쌍별 비교 | 온라인 | 높음 |
| **RLAIF** | AI 기반 | 필요 | 선택적 | AI 생성 | 오프라인/온라인 | 중간 |

### 4.6 미해결 연구 과제

1. **보상 해킹의 근본적 해결**: 프록시 보상과 실제 인간 선호도 간의 괴리를 해소하는 이론적 프레임워크가 부재하다.
2. **다차원 선호도 모델링**: 현재 대부분의 기법은 단일 스칼라 보상을 가정하지만, 인간 선호도는 정확성, 유용성, 안전성 등 다차원적이다. Wang et al. (2024)은 다차원 보상 모델을 탐구하였으나, 실용적 수준의 해결에는 이르지 못하였다.
3. **개인화된 정렬**: 인간 선호도는 개인마다 상이하며, 단일 보상 함수로 모든 사용자의 선호를 반영하는 것은 원리적으로 불가능하다. RLHF의 사회적 선택 이론(social choice theory)적 관점에서의 분석이 필요하다 (Siththaranjan et al., 2024).
4. **확장 가능한 감독(Scalable Oversight)**: 모델이 인간보다 우수한 능력을 갖추게 되는 경우, 인간 피드백의 신뢰성이 저하되는 문제가 발생한다. 이는 약한 감독자로 강한 모델을 정렬하는 "weak-to-strong generalization" 문제와 관련된다 (Burns et al., 2023).

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Deep Reinforcement Learning from Human Preferences | Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, Dario Amodei | **NeurIPS 2017** | https://arxiv.org/abs/1706.03741 |
| 2 | Fine-Tuning Language Models from Human Preferences | Daniel Ziegler, Nisan Stiennon, Jeffrey Wu, Tom Brown, Alec Radford, Dario Amodei, Paul Christiano | arXiv 2019 | https://arxiv.org/abs/1909.08593 |
| 3 | Learning to Summarize from Human Feedback | Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano | **NeurIPS 2020** | https://arxiv.org/abs/2009.01325 |
| 4 | Training language models to follow instructions with human feedback (InstructGPT) | Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, et al. | **NeurIPS 2022** | https://arxiv.org/abs/2203.02155 |
| 5 | Proximal Policy Optimization Algorithms | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov | arXiv 2017 | https://arxiv.org/abs/1707.06347 |
| 6 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher Manning, Chelsea Finn | **NeurIPS 2023** | https://arxiv.org/abs/2305.18290 |
| 7 | A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO) | Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Remi Munos | **AISTATS 2024** | https://arxiv.org/abs/2310.12036 |
| 8 | KTO: Model Alignment as Prospect Theoretic Optimization | Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela | **ICML 2024** | https://arxiv.org/abs/2402.01306 |
| 9 | ORPO: Monolithic Preference Optimization without Reference Model | Jiwoo Hong, Noah Lee, James Thorne | **EMNLP 2024** | https://arxiv.org/abs/2403.07691 |
| 10 | SimPO: Simple Preference Optimization with a Reference-Free Reward | Yu Meng, Mengzhou Xia, Danqi Chen | arXiv 2024 | https://arxiv.org/abs/2405.14734 |
| 11 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo | arXiv 2024 | https://arxiv.org/abs/2402.03300 |
| 12 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek AI | arXiv 2025 | https://arxiv.org/abs/2501.12948 |
| 13 | Scaling Laws for Reward Model Overoptimization | Leo Gao, John Schulman, Jacob Hilton | **ICML 2023** | https://arxiv.org/abs/2210.10760 |
| 14 | RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash | **ICML 2024** | https://arxiv.org/abs/2309.00267 |
| 15 | Constitutional AI: Harmlessness from AI Feedback | Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. | arXiv 2022 | https://arxiv.org/abs/2212.08073 |
| 16 | Training a Helpful and Harmless Assistant with RLHF | Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. | arXiv 2022 | https://arxiv.org/abs/2204.05862 |
| 17 | Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN) | Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu | **ICML 2024** | https://arxiv.org/abs/2401.01335 |
| 18 | Statistical Rejection Sampling Improves Preference Optimization (RSO) | Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, Jialu Liu | **ICLR 2024** | https://arxiv.org/abs/2309.06657 |
| 19 | SLiC-HF: Sequence Likelihood Calibration with Human Feedback | Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, Peter J. Liu | arXiv 2023 | https://arxiv.org/abs/2305.10425 |
| 20 | Secrets of RLHF in Large Language Models Part I: PPO | Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. | arXiv 2023 | https://arxiv.org/abs/2307.04964 |
| 21 | Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study | Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, Yi Wu | **NeurIPS 2024** | https://arxiv.org/abs/2404.10719 |
| 22 | Direct Language Model Alignment from Online AI Feedback | Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, Johan Ferret, Mathieu Blondel | arXiv 2024 | https://arxiv.org/abs/2402.04792 |
| 23 | On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? | Emily Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell | **FAccT 2021** | https://dl.acm.org/doi/10.1145/3442188.3445922 |
| 24 | The Curse of Recursion: Training on Generated Data Makes Models Forget | Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson | arXiv 2024 | https://arxiv.org/abs/2305.17493 |
| 25 | Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision | Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt | arXiv 2023 | https://arxiv.org/abs/2312.09390 |
| 26 | RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment | Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, Tong Zhang | **TMLR 2023** | https://arxiv.org/abs/2304.06767 |
| 27 | Preference Ranking Optimization for Human Alignment (PRO) | Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, Houfeng Wang | **AAAI 2024** | https://arxiv.org/abs/2306.17492 |
| 28 | Reinforcement Learning from Human Feedback with AI Feedback (Anthropic HH-RLHF dataset) | Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, et al. | arXiv 2022 | https://arxiv.org/abs/2209.07858 |
| 29 | Zephyr: Direct Distillation of LM Alignment | Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clementine Fourrier, et al. | arXiv 2023 | https://arxiv.org/abs/2310.16944 |
| 30 | Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons (Bradley-Terry Model) | Ralph Allan Bradley, Milton E. Terry | **Biometrika 1952** | https://doi.org/10.1093/biomet/39.3-4.324 |
| 31 | Prospect Theory: An Analysis of Decision under Risk | Daniel Kahneman, Amos Tversky | **Econometrica 1979** | https://doi.org/10.2307/1914185 |
| 32 | Self-Rewarding Language Models | Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston | arXiv 2024 | https://arxiv.org/abs/2401.10020 |
| 33 | Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint | Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, Tong Zhang | **ICML 2024** | https://arxiv.org/abs/2312.11456 |
| 34 | Reward Model Ensembles Help Mitigate Overoptimization | Thomas Coste, Usman Anwar, Robert Kirk, David Krueger | **ICLR 2024** | https://arxiv.org/abs/2310.02743 |
| 35 | How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation (Social Choice for AI Alignment) | Anikait Siththaranjan, Cassidy Laidlaw, Dylan Hadfield-Menell | arXiv 2024 | https://arxiv.org/abs/2310.10692 |
