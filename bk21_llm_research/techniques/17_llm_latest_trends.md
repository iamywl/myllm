# 2024-2025 주요 학회 LLM 연구 동향 조사

본 문서는 2024-2025년 AI/NLP 탑 컨퍼런스(NeurIPS 2024, ICML 2024, ICLR 2024/2025, ACL 2024, EMNLP 2024, AAAI 2025, CVPR 2024)에서 발표된 LLM 관련 핵심 논문 50편을 주제별로 분석한 것이다. 각 절은 해당 분야의 기법 등장 배경을 문제-해결-새로운 문제의 흐름으로 기술하고, 개별 연구를 통합적 서사 속에서 논의한다.

---

## 1. Long Context LLMs (장문맥 처리)

### 1.1 기법 등장 배경

Transformer 기반 LLM의 자기 어텐션(self-attention)은 시퀀스 길이 $n$에 대하여 $O(n^2)$의 시간 및 공간 복잡도를 갖는다. GPT-3(2020)의 문맥 창은 2,048 토큰, GPT-4(2023)는 8,192~32,768 토큰이었으나, 법률 문서 분석(수만 페이지), 코드베이스 이해(수십만 줄), 장편 소설 요약 등 실제 응용은 수십만~수백만 토큰의 입력을 요구한다. 이 간극을 해소하기 위한 연구는 세 가지 축으로 전개되었다.

**첫째, 위치 인코딩 확장 문제이다.** 사전 학습 시 사용된 위치 인코딩은 학습 길이를 초과하면 외삽(extrapolation) 성능이 급격히 저하된다. RoPE(Rotary Position Embedding)는 상대 위치를 회전 행렬로 인코딩하며, 위치 $m$과 차원 $d$에 대한 주파수 기저는 다음과 같이 정의된다:

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

위치 보간(position interpolation)은 주파수를 스케일링 팩터 $s$로 조정하여 $\theta'_i = \theta_i / s$로 변환함으로써 더 긴 시퀀스를 기존 학습 범위 내에 사상하는 기법이다. 그러나 균일 보간은 고주파 성분의 정보 손실을 유발한다는 한계가 존재하였다.

Ding et al. (Microsoft, ICML 2024)은 **LongRoPE**에서 이 비균일성을 체계적으로 분석하였다 [3]. 핵심 발견은 RoPE의 각 주파수 차원이 위치 보간에 대하여 서로 다른 민감도를 갖는다는 것이다. 이에 따라 차원별로 상이한 스케일링 팩터 $\lambda_i$를 적용하는 비균일 보간을 제안하였다:

$$\theta'_i = \theta_i / \lambda_i, \quad \lambda_i \in [\lambda_{\min}, \lambda_{\max}]$$

진화 탐색(evolutionary search)으로 최적 $\lambda_i$ 조합을 탐색하고, 점진적 확장 전략(progressive extension)을 도입하여 256k 학습 길이에서 1k 미세조정 스텝만으로 문맥 창을 2,048k 토큰까지 확장하였다. 원래의 짧은 문맥(4k)에서도 성능 저하가 0.5% 미만으로 유지된다.

**둘째, KV 캐시 메모리 병목 문제이다.** 자기 회귀 생성 시 모든 이전 토큰의 Key-Value 벡터를 캐시에 유지해야 하므로, 시퀀스 길이에 비례하여 GPU 메모리 소비가 증가한다. LLaMA-7B 기준으로 100만 토큰의 KV 캐시는 FP16에서 약 140GB를 차지하여 단일 GPU에서 처리가 불가능하다. Hooper et al. (UC Berkeley, NeurIPS 2024)은 **KVQuant**에서 KV 캐시 활성화의 저정밀도 양자화를 통하여 이 병목을 해결하였다 [2]. Per-channel key 양자화, pre-RoPE key 양자화, non-uniform quantization, dense-and-sparse 양자화 기법을 조합하여 LLaMA-7B를 단일 A100-80GB GPU에서 100만 토큰, 8-GPU 시스템에서 1,000만 토큰까지 처리 가능하게 하였다. 2비트 양자화에서도 WikiText-2 퍼플렉시티 증가가 0.1 미만이다.

**셋째, 어텐션 연산 자체의 비효율성 문제이다.** 장문맥에서 어텐션 행렬은 대부분 희소(sparse)하지만, 밀집 연산을 수행하므로 불필요한 계산이 발생한다. Jiang et al. (Microsoft, NeurIPS 2024 Spotlight)은 **MInference 1.0**에서 장문맥 어텐션 행렬의 동적 희소 패턴을 체계적으로 분류하였다 [1]. A-shape(초기 토큰 집중), Vertical-Slash(특정 열 집중), Block-Sparse(국소 블록 집중)의 세 가지 고유 패턴을 식별하고, 각 헤드의 패턴을 오프라인에서 결정한 후 GPU 커널 수준에서 해당 패턴에 최적화된 희소 연산을 수행한다. 사전 학습된 LLM에 어떠한 수정도 없이 적용 가능하며, A100에서 pre-filling 지연 시간을 최대 10배 감소시키면서 정확도를 유지한다. 1M 토큰 문맥에서 FlashAttention-2 대비 5.4배 속도 향상을 달성하였다.

이러한 효율성 기법과 병행하여, 장문맥의 실제 활용을 위한 정렬 및 평가 연구도 진행되었다. Ye et al. (ACL 2024)은 **병렬 문맥 인코딩(Parallel Context Encoding)**을 통하여 긴 시퀀스를 청크로 분할하고 병렬로 인코딩함으로써 순차적 처리의 속도 한계를 극복하였다 [4]. Bai et al. (EMNLP 2024 Findings)은 **LongAlign**에서 장문맥 지시 따르기(instruction following) 데이터 구축, 패킹 학습, 손실 가중치 전략을 포함한 체계적인 정렬 파이프라인을 제시하여 장문맥에서의 지시 따르기 성능을 개선하였다 [5]. Li et al. (EMNLP 2024)은 **저차원 투영 어텐션(Low-dimensional Projected Attention, LPA)**을 통하여 어텐션 행렬을 저차원 공간으로 투영하는 방법을 제안하였다 [6]. 쿼리와 키를 $d_k$에서 $d_p$ ($d_p \ll d_k$)로 투영하여 연산량을 절감하며, 학습 시간을 최대 12.4% 절감하면서 테스트 퍼플렉시티를 약 5% 개선하는 성과를 보였다.

장문맥 LLM의 실제 능력을 엄밀히 평가하기 위한 벤치마크도 중요한 연구 방향이다. Kuratov et al. (NeurIPS 2024)은 **BABILong**에서 자연어 도서 텍스트 내에 사실 연쇄(fact chaining), 유도(induction), 공제(deduction) 등 20종의 기초 추론 과제를 삽입하여 LLM의 장문맥 추론 능력을 체계적으로 평가하는 프레임워크를 구축하였다 [7]. 실험 결과 GPT-4와 같은 최신 모델도 10만 토큰 이상에서 정확도가 급락하며, RAG 시스템에서는 시간적 의존성을 요구하는 과제에서 특히 취약함을 보였다.

---

## 2. LLM Reasoning (추론 능력)

### 2.1 기법 등장 배경

Chain-of-Thought(CoT) 프롬프팅(Wei et al., NeurIPS 2022)은 LLM이 중간 추론 단계를 자연어로 명시하도록 유도함으로써 수학 및 논리 추론 성능을 비약적으로 향상시켰다. 그러나 CoT에는 세 가지 근본적 한계가 존재하였다.

**첫째, 토큰 공간의 제약이다.** CoT는 추론의 모든 중간 단계를 이산적 토큰 시퀀스로 직렬화해야 하므로, (1) 토큰 생성 비용이 추론 깊이에 비례하여 증가하고, (2) 자연어로 표현하기 어려운 잠재적 추론 경로(예: 백트래킹, 병렬 가설 평가)를 수행할 수 없다. **둘째, 자기 개선의 한계이다.** LLM이 자체 생성한 CoT의 정확성을 자율적으로 검증하기 어려우며, 외부 검증 신호 없이는 오류가 전파된다. **셋째, 패턴 매칭 의존성이다.** CoT가 진정한 추론인지, 학습 데이터에서 유사 패턴을 검색하는 것인지에 대한 근본적 의문이 제기되었다.

이러한 한계를 극복하기 위한 연구는 크게 세 방향으로 전개되었다.

**잠재 공간 추론(Latent Space Reasoning):** Hao et al. (Meta/FAIR, ICLR 2025)은 **COCONUT(Chain of Continuous Thought)**에서 추론을 토큰 공간이 아닌 연속 잠재 공간에서 수행하는 패러다임을 제안하였다 [8]. LLM의 마지막 은닉 상태 $\mathbf{h}_t \in \mathbb{R}^d$를 "continuous thought"로 활용하며, 이를 다음 추론 단계의 입력으로 직접 피드한다. 학습은 CoT 경로에서 점진적으로 언어 토큰을 연속 사고로 대체하는 커리큘럼 방식으로 진행된다:

$$\text{Stage } k: \quad x_1, \ldots, x_m, \underbrace{\mathbf{h}_1, \ldots, \mathbf{h}_k}_{\text{continuous thoughts}}, t_{k+1}, \ldots, t_n, y$$

핵심 발견은 연속 사고가 다수의 잠재적 다음 추론 단계를 동시에 인코딩하여 너비 우선 탐색(BFS)을 암묵적으로 수행할 수 있다는 것이다. ProntoQA 논리 추론 과제에서 CoT 대비 추론 토큰을 $\sim$50% 절감하면서 정확도를 99.6%로 향상시켰다.

**트리 탐색 기반 자기 학습(Tree-Search Self-Training):** 추론의 자기 개선을 위하여 과정 보상(process reward)과 트리 탐색을 결합하는 접근이 등장하였다. Zhang et al. (Tsinghua University, NeurIPS 2024)은 **ReST-MCTS\***에서 MCTS(Monte Carlo Tree Search) 알고리즘을 LLM 추론에 적용하였다 [9]. 각 추론 단계를 트리의 노드로 모델링하고, 단계별 가치 $V(s)$를 학습하여 탐색을 안내한다. MCTS의 UCB(Upper Confidence Bound) 기반 노드 선택은 다음과 같이 정의된다:

$$\text{UCB}(s) = \frac{Q(s)}{N(s)} + c \cdot \sqrt{\frac{\ln N(\text{parent}(s))}{N(s)}}$$

여기서 $Q(s)$는 누적 보상, $N(s)$는 방문 횟수, $c$는 탐색-활용 균형 상수이다. 수집된 고품질 추론 경로로 정책 모델과 보상 모델을 동시에 갱신하는 강화 자기 학습 루프를 구성하며, MATH 벤치마크에서 기존 best-of-N 대비 3.2% 정확도 향상을 달성하였다.

동일한 자기 개선 방향에서, Tian et al. (NeurIPS 2024)은 **ISC(Imagination, Searching, and Criticizing)** 프레임워크를 제안하여 LLM이 상상(가설 생성), 탐색(해공간 탐색), 비평(자기 검증)의 과정을 통하여 외부 피드백 없이도 자체적으로 추론 능력을 향상시키는 방법론을 제시하였다 [14].

**자기 대전 학습(Self-Play Learning):** Cheng et al. (NeurIPS 2024)은 **SPAG(Self-Playing Adversarial Language Game)**에서 2인 적대적 언어 게임을 통한 추론 능력 향상을 제안하였다 [10]. 공격자와 방어자 역할을 하나의 LLM이 번갈아 수행하며, 게임 결과에 대한 강화 학습(REINFORCE)을 통하여 전략적 추론 능력을 학습한다. 특히 Taboo 게임에서의 자기 대전 학습이 GSM8K, MATH 등 광범위한 추론 벤치마크에서 성능을 일관적으로 향상시킴을 보였으며, LLaMA-2-7B에서 GSM8K 정확도가 5.1% 향상되었다.

**추론 능력의 근본적 한계 분석:** 한편, LLM 추론의 본질에 대한 비판적 연구도 중요한 기여를 하였다. Mirzadeh et al. (Apple, ICLR 2025)은 **GSM-Symbolic**에서 GSM8K 문제의 숫자값, 변수명, 구조를 기호적으로 변환한 변형 인스턴스를 생성하여 LLM의 수학적 추론을 체계적으로 평가하였다 [12]. 핵심 발견은 동일 문제의 서로 다른 인스턴스에서 모델이 최대 10% 이상의 성능 분산을 보이며, 문제에 무관한 정보를 추가하면 정확도가 최대 65%까지 하락한다는 것이다. 이는 현재 LLM의 추론이 형식적 논리보다 패턴 매칭에 가까울 수 있음을 시사한다.

**추론 데이터셋 구축:** 추론 능력의 학습을 위한 고품질 데이터셋 구축도 중요한 연구 방향이다. Toshniwal et al. (NVIDIA, NeurIPS 2024)은 **OpenMathInstruct-1**에서 Mixtral 모델을 사용하여 GSM8K 및 MATH에 대한 코드 인터프리터 기반 솔루션을 합성함으로써 180만 규모의 수학 지시 튜닝 데이터셋을 구축하였다 [11]. 최선 모델인 OpenMath-CodeLlama-70B가 GSM8K에서 84.6%, MATH에서 50.7%를 달성하였다. Wang et al. (NeurIPS 2024)은 **MATHPILE**에서 교과서, 위키피디아, ArXiv 논문 등에서 수집한 수십억 토큰 규모의 수학 사전 학습 코퍼스를 구축하여 수학적 추론의 기반을 강화하였다 [13].

**다국어 코드 추론:** Paul et al. (ACL 2024, Best Paper 후보)은 **IRCoder**에서 컴파일러 중간 표현(IR, Intermediate Representation)을 활용하여 LLM의 다국어 코드 생성 능력을 강건하게 만드는 기법을 제안하였다 [15]. LLVM IR을 매개로 Python, C++, Java 등 다양한 프로그래밍 언어 간 전이 학습을 수행하며, 소스 코드 → IR → 타겟 코드의 2단계 변환이 직접 번역 대비 CodeBLEU 점수를 평균 12.3% 향상시켰다.

---

## 3. LLM Safety and Alignment (안전성 및 정렬)

### 3.1 기법 등장 배경

LLM의 인간 가치 정렬(alignment)은 RLHF(Reinforcement Learning from Human Feedback)를 통하여 달성되어 왔으나, 이 과정에서 세 가지 계열의 문제가 연쇄적으로 발생하였다.

**첫째, RLHF의 구조적 불안정성이다.** 표준 RLHF 파이프라인은 (1) 보상 모델(RM) 학습, (2) PPO 기반 정책 최적화의 2단계로 구성되며, 보상 모델이 유한 데이터에서 학습되므로 학습 분포 밖의 응답에 대하여 부정확한 보상을 할당하는 과최적화(reward hacking) 문제가 발생한다. RLHF의 목적 함수는 다음과 같이 정의된다:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$$

여기서 $r_\phi$는 학습된 보상 모델, $\pi_{\text{ref}}$는 참조 정책, $\beta$는 KL 페널티 계수이다.

이 불안정성을 해결하기 위하여 Rafailov et al. (NeurIPS 2023)의 **DPO(Direct Preference Optimization)**가 등장하여 보상 모델 없이 직접 선호 최적화를 수행하는 접근이 주류가 되었다. DPO 손실 함수는 다음과 같다:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

여기서 $y_w$는 선호 응답, $y_l$은 비선호 응답, $\sigma$는 시그모이드 함수이다.

**둘째, DPO 자체의 한계이다.** 표준 DPO는 승패 간 선호 강도(preference margin)를 반영하지 않고 이진 비교만 수행한다는 문제가 있었다. Amini et al. (ACL 2024 Findings)은 **ODPO(DPO with Offset)**에서 선호 정도를 오프셋 $\delta$로 반영하는 수정을 제안하였다 [24]:

$$\mathcal{L}_{\text{ODPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \delta(x, y_w, y_l) \right) \right]$$

오프셋 $\delta$는 보상 모델 점수 차이 또는 인간 평가 점수 차이에서 산출되며, AlpacaEval에서 표준 DPO 대비 승률이 3.8% 향상되었다. Morimura et al. (EMNLP 2024)은 **fDPO(Filtered DPO)**에서 선호 데이터셋 내 저품질 쌍을 학습된 보상 모델로 필터링하여 DPO의 데이터 품질 문제를 해결하였다 [23]. Lin et al. (EMNLP 2024)은 RLHF 정렬 과정에서 발생하는 일반 능력 저하, 즉 **정렬 세금(alignment tax)** 문제를 분석하여, DPO가 PPO 등 다른 RLHF 알고리즘 대비 정렬 세금을 적게 유발함을 발견하고 이를 완화하는 모델 병합 전략을 제시하였다 [25].

**셋째, 안전 정렬의 취약성이다.** RLHF/DPO로 정렬된 모델이 탈옥(jailbreak) 공격에 취약하다는 사실이 반복적으로 밝혀졌다. 이에 따라 공격 연구와 방어 연구가 동시에 발전하였다.

**공격 측면:** Doumbouya et al. (Stanford, ICLR 2025)은 **h4rm3l**에서 조합 가능한 탈옥 공격을 합성하기 위한 도메인 특화 언어(DSL)를 제안하였다 [18]. 기존 탈옥 공격이 개별적이고 재현 불가능한 반면, h4rm3l은 문자 변환, 역할 주입, 페이로드 분할 등의 원자적 변환을 파이프라인으로 조합할 수 있다. 자동 레드팀 탐색을 통하여 Claude-3-Sonnet과 GPT-4o에 대하여 90% 이상의 공격 성공률(ASR)을 달성하는 공격 조합을 발견하였다. Andriushchenko et al. (ICLR 2025)은 **AgentHarm**에서 LLM 에이전트의 도구 사용 맥락에서의 안전성을 평가하기 위하여 사기, 사이버 범죄, 괴롭힘 등 11개 유해 범주에 걸쳐 110개(증강 포함 440개)의 명시적 악성 에이전트 과제를 구축하였다 [17].

**방어 측면:** Li et al. (ICLR 2025)은 **Safety Layers** 연구에서 정렬된 LLM 내부에서 악의적 질의를 식별하는 데 핵심적 역할을 하는 소수의 연속 계층("안전 계층")을 파라미터 수준에서 발견하였다 [16]. 구체적으로, LLaMA-2-7B-Chat에서 14~20번째 계층의 파라미터 변화가 안전성의 95% 이상을 설명하며, 미세조정 시 이 안전 계층의 파라미터를 동결하면 다운스트림 성능 저하 없이 보안을 유지할 수 있다.

**보상 설계의 혁신:** Mu et al. (NeurIPS 2024)은 **규칙 기반 보상(Rule Based Rewards)**에서 모델 응답의 세부 사양을 명시적 규칙으로 분해하여 원하는 행동과 원하지 않는 행동을 구체적으로 기술하는 기법을 제안하였다 [19]. 기존 RLHF의 모호한 선호 신호를 구조화된 규칙 체계로 대체함으로써 보상 신호의 해석 가능성과 제어 가능성을 동시에 향상시켰다.

**과최적화 완화:** Su et al. (NeurIPS 2024)은 **E-RLHF(Mission Impossible)**에서 RLHF 목적 함수에 안전한 응답의 가능도를 명시적으로 증가시키는 항을 추가하는 간단한 수정을 도입하여 탈옥 공격에 대한 통계적 방어를 제시하였다 [20]. Azar et al. (NeurIPS 2024)은 SFT 손실이 암묵적으로 낙관적 보상 모델(optimistic reward model)로 기능함을 증명하고, 이를 활용하여 RLHF 과최적화를 증명 가능하게 완화하는 **SFT-regularized RLHF**를 제안하였다 [22]. Lee et al. (NeurIPS 2024)은 **One-Shot Safety Alignment**에서 제약 조건부 RLHF를 최적 이중화(optimal dualization)로 풀어 비반복적(one-shot) 안전 정렬을 달성하는 기법을 제시하였다 [21]. 정렬 대상 LM의 안전 속성이 참조 LM을 지정된 마진 $\epsilon$만큼 초과하도록 요구하는 제약을 라그랑지안으로 변환한다:

$$\min_\theta \max_{\lambda \geq 0} \left[ \mathcal{L}_{\text{DPO}}(\theta) + \lambda \left( \epsilon - \mathbb{E}[S_{\text{safety}}(\pi_\theta)] \right) \right]$$

---

## 4. Multimodal LLMs (멀티모달 대형 언어 모델)

### 4.1 기법 등장 배경

대형 언어 모델이 텍스트 영역에서 강력한 범용 능력을 보이면서, 이를 시각, 음성, 비디오 등 다른 모달리티로 확장하려는 연구가 급속히 전개되었다. 멀티모달 LLM(MLLM)의 표준 아키텍처는 시각 인코더(예: CLIP ViT) → 프로젝션 레이어 → LLM 백본의 3단 구조이며, 시각 입력을 토큰 시퀀스로 변환하여 언어 모델에 입력한다. 그러나 이 구조에는 세 가지 핵심 문제가 존재하였다.

**첫째, 시각 토큰 수의 폭발이다.** 고해상도 이미지를 패치 단위로 토큰화하면 단일 이미지에서 수백~수천 개의 시각 토큰이 생성되어, 언어 모델의 문맥 창을 빠르게 소진한다. 비디오의 경우 프레임 수에 비례하여 문제가 심화된다. **둘째, 시각 인코더의 표현 한계이다.** 단일 시각 인코더(예: CLIP ViT-L)는 의미적 표현에는 강하나, 공간적 세부 정보(OCR, 객체 위치, 세밀한 질감)에서 한계를 보인다. **셋째, 과제 통합의 어려움이다.** 이미지 이해, 비디오 이해, 다중 이미지 비교, 객체 접지 등 다양한 비전 과제를 단일 모델로 통합하기 어렵다.

**시각 인코더 설계 공간 탐색:** Tong et al. (NYU, NeurIPS 2024)은 **Cambrian-1**에서 MLLM의 시각 인코더 설계 공간을 체계적으로 탐구하였다 [26]. CLIP, SigLIP, DINOv2, SAM 등 20종 이상의 시각 인코더를 평가하고, 다수의 인코더를 혼합하는 Spatial Vision Aggregator(SVA)를 제안하였다. SVA는 다수 인코더의 특징맵을 동적으로 가중 합산하여 의미적 정보와 공간적 정보를 동시에 포착한다. MLLM 지시 튜닝을 시각 표현의 평가 프로토콜로 활용하는 방법론적 기여도 포함한다.

**통합 멀티모달 모델:** Li et al. (NeurIPS 2024)은 **LLaVA-OneVision**에서 단일 이미지, 다중 이미지, 비디오의 세 가지 시나리오에서 동시에 최고 성능을 달성한 최초의 단일 개방형 MLLM을 구축하였다 [27]. 핵심 전략은 AnyRes(임의 해상도) 기법으로, 입력 이미지를 동적으로 타일링하여 각 타일을 독립적으로 인코딩한 후 합산한다. 단일 이미지 벤치마크에서 GPT-4V를 능가하며 GPT-4o 수준에 근접하는 성능을 보인다. 비디오 이해에서도 32프레임 입력으로 VideoChatGPT 벤치마크에서 SOTA를 달성하였다. Xiao et al. (Microsoft Azure AI, CVPR 2024)은 **Florence-2**에서 1.26억 이미지의 FLD-5B 데이터셋으로 사전 학습된 기반 VLM을 구축하여, 시퀀스-투-시퀀스 아키텍처로 다양한 비전 과제를 통합하였다 [28]. 과제 비특정 제로샷 성능에서 COCO captioning CIDEr 138.0, RefCOCO 정확도 83.6%를 달성하였다.

**언어-객체 접지(Grounding):** Zhang et al. (CVPR 2024)은 **GROUNDHOG**에서 MLLM이 인과 언어 모델링을 통하여 언어-객체 접지를 학습하는 기법을 제안하였다 [29]. 마스크 출력 토큰을 언어 모델의 어휘에 추가하고, 텍스트 생성과 동시에 픽셀 수준의 접지를 수행한다. 과제 특화 미세조정 없이도 refCOCO, refCOCO+, refCOCOg에서 경쟁력 있는 성능을 달성하며, 객체 환각(hallucination)을 크게 감소시킨다.

**장비디오 이해:** 비디오 길이의 확장은 시각 토큰 폭발 문제를 극단적으로 심화시킨다. Song et al. (CVPR 2024)은 **MovieChat**에서 밀집 토큰을 희소 메모리로 변환하는 기법을 통하여 기존 $\sim$100 프레임의 한계를 극복하고 10,000 프레임 이상의 비디오를 처리할 수 있는 장비디오 이해 모델을 구축하였다 [30]. 단기 메모리(최근 프레임)와 장기 메모리(통합 특징)를 분리하여 관리한다. He et al. (CVPR 2024)은 **MA-LMM(Memory-Augmented Large Multimodal Model)**에서 온라인 비디오 스트림을 처리할 수 있는 메모리 증강 구조를 도입하여, 시간 경과에 따른 메모리 뱅크를 자동으로 갱신하며 장기 비디오의 시간적 동적 변화를 효과적으로 파악하였다 [31].

**효율성 및 범용성:** Wu et al. (NeurIPS 2024)은 **VisionLLM v2**에서 시각 인식, 생성, 분할, 편집을 포함하는 수백 종의 비전-언어 과제를 통합된 종단간 프레임워크에서 처리하는 범용 MLLM을 제안하였다 [32]. Li et al. (ICLR 2025)은 **Dynamic-LLaVA**에서 이미지 복잡도에 기반하여 시각 토큰 수를 동적으로 결정하는 희소화 기법을 제안하였다 [34]. 간단한 이미지에서는 시각 토큰의 70% 이상을 제거하면서 성능 저하를 1% 미만으로 유지하여, 추론 FLOPs를 평균 2.3배 절감하였다.

**실세계 평가:** Zhang et al. (ICLR 2025)은 **MME-RealWorld**에서 기존 합성 데이터 기반 벤치마크의 한계를 극복하고 실제 환경 사진에서 수집한 과제로 MLLM을 평가하는 벤치마크를 구축하였다 [33]. 43개국에서 수집된 29,429개 QA 쌍을 포함하며, 최고 성능 모델도 60% 미만의 정확도를 보여 실세계 시나리오에서의 격차를 정량화하였다.

---

## 5. LLM Agents (에이전트)

### 5.1 기법 등장 배경

LLM이 자연어 이해와 생성에서 인간 수준에 근접하면서, 이를 실제 환경에서 자율적으로 과제를 수행하는 에이전트로 활용하려는 연구가 급속히 확대되었다. LLM 에이전트의 표준 구조는 ReAct(Yao et al., ICLR 2023)에서 제시된 Thought-Action-Observation 루프를 따르며, LLM이 상황을 분석(Thought)하고 도구를 호출(Action)한 후 결과를 관찰(Observation)하는 과정을 반복한다. 그러나 이 구조에는 세 가지 핵심 과제가 존재하였다.

**첫째, 에이전트-환경 인터페이스의 설계이다.** LLM은 자연어를 처리하도록 학습되었으나, 실제 환경(터미널, 브라우저, GUI)은 비정형적 출력을 생성한다. 에이전트가 환경 상태를 정확히 인식하고 적절한 행동을 실행하려면, 환경 출력을 LLM이 이해할 수 있는 형태로 변환하고 LLM의 출력을 환경이 실행할 수 있는 명령으로 변환하는 인터페이스가 필수적이다. **둘째, 장기 계획 및 오류 복구의 어려움이다.** 실제 과제는 수십 단계의 행동 시퀀스를 요구하며, 초기 단계의 오류가 후속 단계로 전파된다. **셋째, 평가의 표준화 부재이다.** 에이전트 능력을 재현 가능하게 측정할 수 있는 벤치마크가 부족하였다.

**에이전트-컴퓨터 인터페이스(ACI):** Yang et al. (Princeton, NeurIPS 2024)은 **SWE-agent**에서 LM 에이전트가 소프트웨어 엔지니어링 과제를 해결하기 위한 맞춤형 에이전트-컴퓨터 인터페이스(ACI)를 설계하였다 [35]. 핵심 설계 원칙은 (1) 행동을 간결하고 직관적으로 구성하고, (2) 환경 피드백을 정보적이면서도 간결하게 제공하며, (3) 오류 시 자동 복구 메커니즘을 포함하는 것이다. 예를 들어, 파일 편집 시 전체 파일이 아닌 변경 부분만을 보여주고, 구문 오류 시 자동으로 원래 상태로 롤백한다. SWE-bench에서 12.5%의 pass@1(이전 SOTA 3.8%), HumanEvalFix에서 87.7%의 pass@1을 달성하였다.

**실세계 벤치마크:** Xie et al. (NeurIPS 2024, Datasets and Benchmarks Track)은 **OSWorld**에서 Ubuntu, Windows, macOS를 지원하는 최초의 확장 가능한 실제 컴퓨터 환경 멀티모달 에이전트 벤치마크를 구축하였다 [36]. 369개 컴퓨터 과제를 포함하며, 파일 관리, 웹 브라우징, 코드 편집, 멀티앱 워크플로우 등을 포괄한다. 인간이 72.36%를 달성하는 반면 최선 모델(GPT-4V)은 12.24%에 그쳐 GUI 접지, 장기 계획, 운영 체제 지식에서의 격차가 정량적으로 확인되었다. Ma et al. (NeurIPS 2024, Datasets and Benchmarks Track)은 **AgentBoard**에서 멀티턴 LLM 에이전트의 체계적 평가를 위한 분석 보드를 제안하여, 에이전트의 계획, 도구 사용, 자기 반성 능력을 분리하여 측정할 수 있는 프레임워크를 제공하였다 [38]. Xu et al. (EMNLP 2024)은 **MAgIC**에서 경쟁 기반 멀티에이전트 환경에서 판단, 추론, 기만, 자기 인식, 협력, 합리성을 정량적으로 평가하는 벤치마크를 구축하였다 [41].

**도구 활용 최적화:** Wu et al. (NeurIPS 2024)은 **AvaTaR**에서 LLM 에이전트의 도구 활용 능력을 자동으로 최적화하는 프레임워크를 제안하였다 [37]. 비교기(comparator) 모듈이 성공/실패 실행 경로를 대조 분석하여 핵심 차이점을 식별하고, 이를 바탕으로 통찰력 있는 프롬프트를 반복적으로 생성한다. 4개 다운스트림 과제에서 기존 프롬프트 대비 평균 7.2% 성능 향상을 달성하였다.

**에이전트 계획 및 협업:** Hao et al. (NeurIPS 2024)은 **World Knowledge Model**을 에이전트 계획에 통합하여, 사전 학습된 LLM의 세계 지식을 상태 전이 예측에 활용하는 접근을 제안하였다 [39]. Zhang et al. (NeurIPS 2024)은 **반성적 멀티에이전트 협업(Reflective Multi-Agent Collaboration)**에서 에이전트 간 비평과 반성을 통한 협업 프레임워크를 제안하여, 단일 에이전트 대비 복잡한 과제에서의 성공률을 향상시켰다 [40]. Feng et al. (EMNLP 2024 Findings)은 **ReHAC**에서 강화 학습 기반 인간-에이전트 협업 방법을 제안하여, 과제 해결 과정에서 인간 개입의 최적 시점을 결정하는 정책 모델을 학습하였다 [42]. 이는 완전 자율 에이전트의 한계를 인정하고, 인간과의 효율적 협업 지점을 학습하는 실용적 접근이다.

---

## 6. Small Language Models / On-Device LLMs (소형 언어 모델)

### 6.1 기법 등장 배경

대형 LLM은 높은 추론 비용(GPT-4 추론 시 수십 ms/토큰, 수만 달러/월), 높은 지연 시간(클라우드 왕복 50-200ms), 개인 정보 보호 문제(데이터의 외부 전송)로 인하여 모바일 기기, 엣지 디바이스, 로컬 서버에서의 실행이 불가능하다. 이에 따라 10억 파라미터 이하에서도 대형 모델에 근접하는 성능을 달성하는 소형 언어 모델(SLM) 연구가 세 가지 방향으로 전개되었다.

**첫째, 데이터 큐레이션이다.** 신경망 스케일링 법칙(Chinchilla scaling law)에 따르면, 모델 파라미터 수 $N$과 학습 토큰 수 $D$의 최적 관계는 다음과 같이 정의된다:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

여기서 $A, B, E$는 상수, $\alpha \approx 0.34$, $\beta \approx 0.28$이다. 소형 모델에서는 $N$이 작으므로, $D$를 크게 늘리고 데이터 품질을 극대화하여 성능을 보상해야 한다.

Abdin et al. (Microsoft, arXiv 2024)은 **Phi-3**에서 이 원리를 극단적으로 적용하였다 [44]. 38억 파라미터의 phi-3-mini를 3.3조 토큰의 고도로 큐레이션된 데이터셋으로 학습하여, Mixtral 8x7B(46.7B 활성 파라미터) 및 GPT-3.5에 필적하는 성능을 달성하였다. 핵심 전략은 (1) 웹 데이터를 LLM으로 필터링하고, (2) 합성 데이터를 생성하며, (3) 교과서 품질의 데이터를 선별하는 것이다. MMLU 69.7%, GSM8K 85.0%, HumanEval 62.2%를 달성하였으며, iPhone 14에서 초당 12 토큰의 생성 속도를 보인다.

**둘째, 아키텍처 최적화이다.** 소형 모델에서는 대형 모델과 상이한 아키텍처 설계 원칙이 필요하다. Liu et al. (Meta, ICML 2024)은 **MobileLLM**에서 10억 파라미터 미만 LLM의 설계 요소를 체계적으로 분석하였다 [43]. 핵심 발견은 소형 모델에서는 넓고 얕은 구조보다 **깊고 얇은(deep-and-thin) 아키텍처**가 우수하다는 것이다. 동일 파라미터 예산에서 계층 수를 2배로 늘리고 은닉 차원을 줄이면 퍼플렉시티가 $\sim$0.5 개선된다. SwiGLU 활성화 함수, 임베딩 공유(weight tying), 그룹 쿼리 어텐션(GQA)을 통합하고, 계층 간 가중치 공유를 도입하여 125M/350M 모델에서 SOTA를 달성하였다.

Kim et al. (NeurIPS 2024)은 **Search for Efficient LLMs**에서 LLM을 위한 신경망 아키텍처 탐색(NAS)을 연구하였다 [46]. 기존 NAS 기법이 LLM의 거대한 탐색 공간과 학습 비용에 적용되기 어려운 문제를 해결하기 위하여, 초네트워크(supernet) 기반 one-shot NAS를 LLM에 적응시키는 방법을 제안하였다.

**셋째, 모델 압축(가지치기 + 지식 증류)이다.** 대형 모델에서 소형 모델로 지식을 전이하는 접근이다. 지식 증류의 핵심 손실 함수는 교사 모델 $T$와 학생 모델 $S$의 출력 분포 간 KL 발산을 최소화한다:

$$\mathcal{L}_{\text{KD}} = \sum_{t} D_{\text{KL}} \left( \text{softmax}(\mathbf{z}_T^{(t)} / \tau) \| \text{softmax}(\mathbf{z}_S^{(t)} / \tau) \right)$$

여기서 $\mathbf{z}_T^{(t)}$, $\mathbf{z}_S^{(t)}$는 각각 교사와 학생의 시점 $t$에서의 로짓, $\tau$는 온도 하이퍼파라미터이다.

Muralidharan et al. (NVIDIA, NeurIPS 2024)은 **Minitron/Compact LMs**에서 깊이, 너비, 어텐션 헤드, MLP 차원의 구조적 가지치기(structured pruning)와 지식 증류 기반 재학습을 결합한 LLM 압축 파이프라인을 체계화하였다 [45]. Nemotron-4 15B를 8B와 4B로 압축하면서 동일 규모 처음부터 학습한 모델 대비 언어 모델링 퍼플렉시티에서 우위를 보이며, 압축에 필요한 학습 토큰이 처음부터 학습 대비 40배 적다.

Jia et al. (NeurIPS 2024)은 **적대적 모멘트 매칭 증류(Adversarial Moment-Matching Distillation)**에서 기존 KL 발산 기반 화이트박스 증류의 한계를 지적하였다 [47]. 교사-학생 분포의 모든 모멘트를 매칭하는 적대적 학습 기반 증류를 제안하여, 교사 분포의 다중 모드(multi-modal) 구조를 보다 충실하게 전달한다.

Riviere et al. (Google DeepMind, arXiv 2024)은 **Gemma 2**에서 2B-27B 파라미터 범위의 개방형 모델 계열을 발표하였다 [48]. 온라인 증류(online distillation)를 도입하여 학습 과정에서 대형 교사 모델의 소프트 레이블을 실시간으로 활용하며, 모바일 GPU에서 최대 2,585 tokens/sec의 pre-fill 속도를 달성한다. Int4 양자화로 모델 크기를 2.5-4배 축소하면서도 MMLU에서 2B 모델이 51.3%, 9B 모델이 71.3%를 달성하여 동급 최강의 성능을 보인다.

---

## 7. 학회별 주요 수상 논문 및 주목할 연구

2024-2025년 주요 학회의 수상 논문 중 LLM과 관련된 연구는 다음과 같다.

Kondratyuk et al. (Google, ICML 2024 Best Paper Award)은 **VideoPoet**에서 대형 언어 모델 아키텍처를 활용한 제로샷 비디오 생성 기법을 제안하였다 [49]. 시각, 오디오, 텍스트 토큰을 통합된 어휘로 토큰화하고, 자기 회귀 트랜스포머로 비디오를 생성한다. 비디오 편집, 스타일 변환, 비디오-음성 합성 등 다양한 생성 과제를 단일 모델로 수행할 수 있다.

Zhang et al. (EMNLP 2024 Best Paper Award)은 **사전 학습 데이터 검출(Pretraining Data Detection)**을 위한 발산 기반 보정(divergence-based calibration) 기법을 제안하였다 [50]. 멤버십 추론 공격(MIA)에서 핵심 과제인 비회원 참조 분포의 부재를 소형 참조 모델과의 발산으로 보정하여 해결하며, WikiMIA 벤치마크에서 기존 SOTA 대비 AUC를 5.3% 향상시켰다.

Panickssery et al. (NeurIPS 2024 Oral)은 **LLM 자기 편향(self-preference bias)** 현상을 발견하였다 [51]. GPT-4, Llama 2 등의 LLM이 자신이 생성한 텍스트와 타 LLM 또는 인간 생성 텍스트를 구별할 수 있으며, 자체 생성물을 체계적으로 선호하는 편향이 있음을 입증하였다. 이는 LLM-as-judge 평가 패러다임의 신뢰성에 중요한 시사점을 제공한다.

---

## 종합 분석

2024-2025년 LLM 연구의 주요 흐름을 종합하면 다음과 같다.

1. **장문맥 처리**: KV 캐시 양자화(KVQuant), 동적 희소 어텐션(MInference), 비균일 위치 보간(LongRoPE)의 세 축이 상호 보완적으로 작용하여, 단일 GPU에서 수백만 토큰 처리가 실용적 수준에 도달하였다. 핵심 수식은 RoPE 주파수 스케일링 $\theta'_i = \theta_i / \lambda_i$와 KV 캐시 양자화에서의 per-channel 정규화이다.

2. **추론 능력**: CoT의 토큰 공간 제약을 극복하는 잠재 공간 추론(COCONUT), 과정 보상 기반 트리 탐색 자기 학습(ReST-MCTS*), 적대적 자기 대전(SPAG)이 새로운 패러다임으로 부상하였다. GSM-Symbolic의 결과는 현재 LLM 추론의 패턴 매칭 의존성이라는 근본적 한계를 시사한다.

3. **안전성 및 정렬**: DPO 변형의 급증(ODPO, fDPO)과 안전 계층의 발견(Safety Layers)이 이론적 기반을 강화하였고, h4rm3l의 조합 가능한 탈옥 공격은 방어 연구의 긴급성을 부각시켰다. 핵심 수식은 DPO 손실과 그 변형(오프셋, 필터링)이다.

4. **멀티모달**: 단일 모델로 이미지/비디오/다중 이미지를 통합 처리하는 방향(LLaVA-OneVision)과 시각 토큰 효율화(Dynamic-LLaVA), 장비디오 이해(MovieChat, MA-LMM)가 핵심이다. 시각 인코더 혼합(Cambrian-1)과 동적 토큰 희소화가 설계 공간의 주요 축이다.

5. **에이전트**: 에이전트-컴퓨터 인터페이스 설계(SWE-agent)가 성능의 핵심 요소로 부상하였으며, 실세계 벤치마크(OSWorld: 인간 72.36% vs 모델 12.24%)가 현재의 격차를 정량화하였다. 멀티에이전트 협업과 인간-에이전트 협업(ReHAC)이 완전 자율 에이전트의 한계를 보완한다.

6. **소형 모델**: 데이터 큐레이션(Phi-3: 38억 파라미터로 GPT-3.5급), 아키텍처 최적화(MobileLLM: 깊고 얇은 구조), 구조적 가지치기+증류(Minitron: 40배 적은 학습 토큰)가 핵심 전략이다. 스케일링 법칙 $L(N,D)$에 기반한 데이터-모델 크기 최적화가 이론적 기반을 제공한다.

---

## 참고 문헌 목록 (전체)

| # | 논문 제목 | 저자 | 학회 | 연도 |
|---|-----------|------|------|------|
| 1 | MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention | Jiang et al. (Microsoft) | NeurIPS (Spotlight) | 2024 |
| 2 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | Hooper et al. (UC Berkeley) | NeurIPS | 2024 |
| 3 | LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | Ding et al. (Microsoft) | ICML | 2024 |
| 4 | Long-Context Language Modeling with Parallel Context Encoding | Ye et al. | ACL | 2024 |
| 5 | LongAlign: A Recipe for Long Context Alignment of Large Language Models | Bai et al. | EMNLP Findings | 2024 |
| 6 | Scalable Efficient Training of LLMs with Low-dimensional Projected Attention | Li et al. | EMNLP | 2024 |
| 7 | BABILong: Testing the Limits of LLMs in Processing Long Contexts | Kuratov et al. | NeurIPS | 2024 |
| 8 | COCONUT: Training Large Language Models to Reason in a Continuous Latent Space | Hao et al. (Meta/FAIR) | ICLR | 2025 |
| 9 | ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search | Zhang et al. (Tsinghua) | NeurIPS | 2024 |
| 10 | Self-Playing Adversarial Language Game Enhances LLM Reasoning | Cheng et al. | NeurIPS | 2024 |
| 11 | OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset | Toshniwal et al. (NVIDIA) | NeurIPS | 2024 |
| 12 | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs | Mirzadeh et al. (Apple) | ICLR | 2025 |
| 13 | MATHPILE: A Billion-Token-Scale Pre-training Corpus for Math | Wang et al. | NeurIPS | 2024 |
| 14 | Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing | Tian et al. | NeurIPS | 2024 |
| 15 | IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators | Paul et al. | ACL (Best Paper 후보) | 2024 |
| 16 | Safety Layers in Aligned Large Language Models: The Key to LLM Security | Li et al. | ICLR | 2025 |
| 17 | AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents | Andriushchenko et al. | ICLR | 2025 |
| 18 | h4rm3l: A Language for Composable Jailbreak Attack Synthesis | Doumbouya et al. (Stanford) | ICLR | 2025 |
| 19 | Rule Based Rewards for Language Model Safety | Mu et al. | NeurIPS | 2024 |
| 20 | Mission Impossible: A Statistical Perspective on Jailbreaking LLMs | Su et al. | NeurIPS | 2024 |
| 21 | One-Shot Safety Alignment for LLMs via Optimal Dualization | Lee et al. | NeurIPS | 2024 |
| 22 | Provably Mitigating Overoptimization in RLHF | Azar et al. | NeurIPS | 2024 |
| 23 | Filtered Direct Preference Optimization (fDPO) | Morimura et al. | EMNLP | 2024 |
| 24 | Direct Preference Optimization with an Offset (ODPO) | Amini et al. | ACL Findings | 2024 |
| 25 | Mitigating the Alignment Tax of RLHF | Lin et al. | EMNLP | 2024 |
| 26 | Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs | Tong et al. (NYU) | NeurIPS | 2024 |
| 27 | LLaVA-OneVision: Easy Visual Task Transfer | Li et al. | NeurIPS | 2024 |
| 28 | Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks | Xiao et al. (Microsoft) | CVPR | 2024 |
| 29 | GROUNDHOG: Grounding Large Language Models to Holistic Segmentation | Zhang et al. | CVPR | 2024 |
| 30 | MovieChat: From Dense Token to Sparse Memory for Long Video Understanding | Song et al. | CVPR | 2024 |
| 31 | MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding | He et al. | CVPR | 2024 |
| 32 | VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model | Wu et al. | NeurIPS | 2024 |
| 33 | MME-RealWorld: A Real-World Benchmark for Multimodal LLMs | Zhang et al. | ICLR | 2025 |
| 34 | Dynamic-LLaVA: Efficient Multimodal LLMs via Dynamic Vision-Language Context Sparsification | Li et al. | ICLR | 2025 |
| 35 | SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering | Yang et al. (Princeton) | NeurIPS | 2024 |
| 36 | OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments | Xie et al. | NeurIPS | 2024 |
| 37 | AvaTaR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning | Wu et al. | NeurIPS | 2024 |
| 38 | AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents | Ma et al. | NeurIPS | 2024 |
| 39 | Agent Planning with World Knowledge Model | Hao et al. | NeurIPS | 2024 |
| 40 | Reflective Multi-Agent Collaboration based on Large Language Models | Zhang et al. | NeurIPS | 2024 |
| 41 | MAgIC: Investigation of LLM Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration | Xu et al. | EMNLP | 2024 |
| 42 | ReHAC: LLM-based Human-Agent Collaboration for Complex Task Solving | Feng et al. | EMNLP Findings | 2024 |
| 43 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases | Liu et al. (Meta) | ICML | 2024 |
| 44 | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | Abdin et al. (Microsoft) | arXiv (MS) | 2024 |
| 45 | Compact Language Models via Pruning and Knowledge Distillation | Muralidharan et al. (NVIDIA) | NeurIPS | 2024 |
| 46 | Search for Efficient Large Language Models | Kim et al. | NeurIPS | 2024 |
| 47 | Adversarial Moment-Matching Distillation of Large Language Models | Jia et al. | NeurIPS | 2024 |
| 48 | Gemma 2: Open Models Based on Gemini Research and Technology | Riviere et al. (Google DeepMind) | arXiv (Google) | 2024 |
| 49 | VideoPoet: A Large Language Model for Zero-Shot Video Generation | Kondratyuk et al. (Google) | ICML (Best Paper) | 2024 |
| 50 | Pretraining Data Detection for LLMs: A Divergence-based Calibration Method | Zhang et al. | EMNLP (Best Paper) | 2024 |
| 51 | LLM Evaluators Recognize and Favor Their Own Generations | Panickssery et al. | NeurIPS (Oral) | 2024 |
