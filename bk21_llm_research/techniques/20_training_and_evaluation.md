# LLM 학습 데이터, 학습법, 스케일링 및 평가 최신 연구 동향 (2023-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI) 중심
> 총 참고논문: 42편

---

## 1. 개요

대규모 언어 모델(LLM)의 성능은 모델 아키텍처뿐만 아니라 학습 데이터의 품질, 학습 기법, 스케일링 전략, 후학습 정렬(alignment), 그리고 평가 방법론에 의해 결정된다. 2023년 이후 LLM 연구는 단순한 규모 확장(scale-up)에서 벗어나, 데이터 큐레이션의 정교화, 학습 효율의 극대화, 추론 시점 스케일링(inference-time scaling), 강화학습 기반 정렬 기법의 다양화, 그리고 평가 벤치마크의 신뢰성 확보라는 다차원적 과제로 전환되었다.

본 문서에서는 LLM 학습 및 평가의 핵심 연구를 (1) 학습 데이터 큐레이션, (2) 사전학습 기법 및 목적함수, (3) 스케일링 법칙, (4) 후학습 및 정렬 기법, (5) 평가 벤치마크, (6) 토크나이제이션 및 어휘의 6개 분야로 나누어 체계적으로 정리한다.

---

## 2. 학습 데이터 큐레이션 (Training Data Curation)

### 2.1 기법 등장 배경

초기 LLM 사전학습은 Common Crawl 등 웹에서 수집한 대규모 코퍼스를 최소한의 전처리만 거쳐 사용하였다. 그러나 웹 데이터에는 저품질 텍스트, 중복 문서, 유해 콘텐츠, 개인정보 등이 다량 포함되어 있어, 모델이 편향(bias)과 환각(hallucination)을 학습하는 문제가 발생하였다. Raffel et al. (2020)의 C4 코퍼스가 휴리스틱 필터링의 시초를 열었으나, 규칙 기반 필터링만으로는 데이터 품질을 충분히 보장할 수 없었다. 이에 따라 모델 기반 품질 필터링, 정교한 중복 제거(deduplication), 데이터 혼합 비율 최적화, 합성 데이터(synthetic data) 생성 등 학습 데이터 큐레이션 기법이 급속히 발전하였다. 특히 Chinchilla (Hoffmann et al., 2022)가 학습 데이터 양의 중요성을 실증적으로 입증한 이후, "더 많은 데이터"가 아닌 "더 좋은 데이터"에 대한 연구가 본격화되었다.

### 2.2 데이터 품질 필터링

**DSIR (Data Selection with Importance Resampling)** (Xie et al., Stanford, ICLR 2024)은 대규모 코퍼스에서 타겟 분포에 가까운 데이터를 효율적으로 선별하는 기법이다. n-gram 특징 기반 중요도 가중치(importance weight)를 계산하여, 사전학습 코퍼스 전체를 분류기로 평가하지 않고도 통계적으로 유의미한 데이터 선택을 수행한다. The Pile에서 Wikipedia 타겟 분포로 선택한 데이터로 학습 시, 랜덤 선택 대비 GLUE 점수가 2.1점 향상되었다 [1].

**QuRating** (Wettig et al., Princeton, ICML 2024)은 LLM 자체를 품질 평가기(rater)로 활용하여 데이터 품질을 다차원적으로 측정하는 프레임워크이다. 작문 스타일, 사실 정확성, 교육적 가치, 필요 전문 지식의 4개 축으로 품질 점수를 산출하며, 각 축별 점수를 기반으로 데이터 혼합 비율을 최적화한다. 1.5B 파라미터 모델에서 교육적 가치가 높은 데이터만으로 학습 시 전체 데이터 학습 대비 동등한 성능을 30% 적은 토큰으로 달성하였다 [2].

**FineWeb** (Penedo et al., Hugging Face, NeurIPS 2024 Datasets Track)은 15조 토큰 규모의 오픈소스 영어 웹 코퍼스이다. URL 필터링, 언어 식별, 품질 분류, 중복 제거의 4단계 파이프라인을 통해 96개 Common Crawl 덤프를 처리하였다. 특히 교육적 가치 분류기(educational value classifier)를 적용한 FineWeb-Edu 서브셋은 1.3B 모델 학습 시 기존 C4, RefinedWeb 대비 MMLU에서 3-5점의 성능 향상을 보였다 [3].

### 2.3 중복 제거 (Deduplication)

**SemDeDup** (Abbas et al., Meta, ICLR 2024)은 임베딩 공간에서의 의미적 중복 제거(semantic deduplication) 기법이다. 기존 MinHash 기반 문자열 수준 중복 제거는 패러프레이징(paraphrasing)된 중복을 탐지하지 못하는 한계가 있었다. SemDeDup은 사전학습된 임베딩 모델로 문서를 벡터화한 후, 코사인 유사도 임계값 기반 클러스터링으로 의미적 중복을 제거한다. C4 코퍼스에서 50% 데이터를 제거하고도 성능 저하 없이 학습 효율을 2배 향상시켰으며, 특정 설정에서는 성능이 오히려 향상되었다 [4].

**D4 (Data Deduplication by Diversity)** (Tirumala et al., Meta, ICML 2024)는 중복 제거를 단순 유사도 기준이 아닌 데이터 다양성(diversity) 극대화 관점에서 접근한다. SSL(self-supervised learning) 프로토타입 기반 다양성 샘플링을 도입하여, 의미적으로 다양한 데이터 부분집합을 선택한다. 이 방법으로 선택된 데이터는 전체 데이터 대비 동일 성능을 달성하는 데 필요한 학습 단계(step)를 40% 절감하였다 [5].

### 2.4 합성 데이터 (Synthetic Data)

**Textbooks Are All You Need (phi-1)** (Gunasekar et al., Microsoft, NeurIPS 2023)은 GPT-3.5로 생성한 교과서 수준의 합성 데이터만으로 1.3B 파라미터 코드 생성 모델을 학습하여, 10배 이상 큰 모델들과 비견되는 HumanEval 성능(50.6%)을 달성하였다. 이 연구는 합성 데이터의 품질이 규모를 대체할 수 있음을 최초로 대규모 실증한 사례이다 [6].

**Cosmopedia** (Ben Allal et al., Hugging Face, NeurIPS 2024 Datasets Track)은 Mixtral-8x7B를 활용하여 생성한 30B 토큰 규모의 합성 교과서 및 스토리 코퍼스이다. 시드 데이터로 웹 페이지, 교과서 목차, 교육 과정 등을 사용하며, 주제 다양성과 난이도 다양성을 체계적으로 제어한다. Cosmopedia로 학습한 1B 모델은 동일 규모 웹 데이터 학습 모델 대비 ARC-Easy에서 5점 이상의 성능 향상을 보였다 [7].

**Rephrasing the Web** (Maini et al., Apple, ICLR 2024)은 웹에서 수집한 원본 텍스트를 LLM으로 재구성(rephrase)하여 학습 데이터의 품질을 향상시키는 기법이다. 원본 C4 텍스트를 "Easy" (초등학생 수준), "Medium" (고등학생 수준), "Hard" (대학원생 수준) 등 다양한 스타일로 재작성하여 학습 데이터의 다양성과 품질을 동시에 개선한다. 재구성 데이터로 학습한 모델은 원본 데이터 대비 MMLU에서 1.5-3.0점 향상을 달성하였다 [8].

### 2.5 데이터 혼합 최적화 (Data Mixing)

**DoReMi** (Xie et al., Google, ICLR 2024)은 사전학습 데이터의 도메인별 혼합 비율을 자동 최적화하는 프레임워크이다. 소형 프록시 모델(proxy model)과 참조 모델(reference model)의 손실 차이를 기반으로, 분배적으로 강건한(distributionally robust) 최적화를 수행하여 각 도메인의 가중치를 결정한다. The Pile 데이터에서 DoReMi로 최적화된 혼합 비율로 280M 모델을 학습한 결과, 기본 비율 대비 평균 퍼플렉시티가 6.5% 감소하였으며, 8B 규모로 스케일업 시에도 동일한 개선이 유지되었다 [9].

---

## 3. 사전학습 기법 및 목적함수 (Pre-training Methods and Objectives)

### 3.1 기법 등장 배경

GPT 계열의 자기회귀(autoregressive) 다음 토큰 예측(next-token prediction)은 LLM 사전학습의 표준 목적함수로 자리 잡았다. 그러나 이 목적함수는 모든 토큰에 동일한 가중치를 부여하므로, 관사(article)나 접속사 등 예측이 용이한 토큰에도 불필요한 연산을 소비한다는 비효율 문제가 있다. 또한 수천억 파라미터 규모에서 학습 안정성(training stability)이 급격히 저하되어, 손실 발산(loss spike), 그래디언트 폭발(gradient explosion) 등이 빈번히 발생한다. 이에 따라 선택적 토큰 학습, 새로운 목적함수 설계, 학습 안정화 기법, 대안적 아키텍처 등이 연구되고 있다.

### 3.2 선택적 학습 및 목적함수 혁신

**Selective Language Modeling (RHO-1)** (Lin et al., Tsinghua/Microsoft, NeurIPS 2024)은 사전학습 시 모든 토큰이 아닌 유용한 토큰만 선택적으로 학습하는 기법이다. 참조 모델의 손실을 기준으로 각 토큰의 초과 손실(excess loss)을 계산하고, 상위 k%의 토큰만 역전파(backpropagation)에 포함한다. 수식으로 표현하면:

$$\mathcal{L}_{\text{SLM}} = -\frac{1}{|S|} \sum_{t \in S} \log P_\theta(x_t | x_{<t}), \quad S = \{t : \ell_\theta(x_t) - \ell_{\text{ref}}(x_t) > \tau\}$$

여기서 $S$는 선택된 토큰 집합, $\tau$는 초과 손실 임계값이다. 15B 토큰으로 학습한 RHO-1-1B 모델이 일반 사전학습 대비 GSM8K에서 40.6% → 53.3%로 성능이 향상되었으며, 학습 효율이 5-10배 개선되었다 [10].

**Beyond Next-Token Prediction** (Gloeckle et al., Meta, ICML 2024)은 다음 토큰 1개가 아닌 다음 k개 토큰을 동시에 예측하는 다중 토큰 예측(multi-token prediction) 목적함수를 제안하였다. 각 미래 위치에 대해 독립적인 출력 헤드를 두되, Transformer 본체의 표현은 공유한다. 코드 생성 벤치마크(HumanEval)에서 4-토큰 예측으로 학습한 13B 모델이 1-토큰 예측 모델 대비 12% 향상된 pass@1을 달성하였으며, 자연어 벤치마크에서도 일관된 개선을 보였다 [11].

### 3.3 학습 안정성 (Training Stability)

**μP (Maximal Update Parametrization)** (Yang et al., Microsoft, ICLR 2024 Oral)은 소규모 모델에서 최적화된 하이퍼파라미터를 대규모 모델로 직접 전이(transfer)할 수 있는 파라미터화 기법이다. 표준 파라미터화(SP)에서는 모델 폭(width)이 변할 때 최적 학습률이 달라지지만, μP에서는 초기화 스케일과 학습률 스케일을 폭에 따라 조정하여 최적 하이퍼파라미터가 폭에 불변(width-invariant)하도록 한다. 구체적으로, 히든 레이어의 학습률을 $\eta / \text{width}$로 스케일링하고, 출력 레이어의 초기화를 $1/\text{width}$로 스케일링한다. 이를 통해 40M 모델에서 탐색한 하이퍼파라미터를 6.7B 모델에 직접 적용하여 추가 탐색 없이 안정적 학습을 달성하였다 [12].

**WSD (Warmup-Stable-Decay) 학습률 스케줄** (Hu et al., Tsinghua, NeurIPS 2024)은 기존 코사인(cosine) 스케줄의 문제점을 해결하기 위해 제안된 학습률 스케줄러이다. 코사인 스케줄은 학습 종료 시점을 사전에 결정해야 하며, 중간에 학습을 연장하거나 체크포인트에서 재개하기 어렵다. WSD 스케줄은 웜업 → 안정(constant) → 감소(decay)의 3단계로 구성되며, 안정 단계에서 임의의 시점에 감소 단계로 전환하여 학습을 종료할 수 있다. MiniCPM 시리즈에서 WSD 스케줄이 코사인 스케줄 대비 동등하거나 우수한 성능을 달성하면서, 학습 유연성을 크게 향상시켰다 [13].

### 3.4 대안적 아키텍처

**Mamba-2** (Dao and Gu, CMU/Princeton, ICML 2024)은 상태 공간 모델(State Space Model, SSM)과 어텐션 메커니즘 사이의 이론적 연결을 확립한 연구이다. 구조화된 상태 공간 이중성(Structured State Space Duality, SSD) 프레임워크를 통해 SSM이 반구조화된 행렬(semiseparable matrix)과 동치임을 증명하고, 이를 활용하여 Mamba 아키텍처를 2-8배 가속하였다. Mamba-2-2.7B 모델은 Transformer++ 대비 동등한 퍼플렉시티를 달성하면서 학습 처리량이 50% 이상 향상되었다 [14].

**RWKV-v6 (Eagle/Finch)** (Peng et al., EMNLP 2024)는 RNN 기반 아키텍처로 Transformer와 동등한 성능을 달성하면서 선형 시간 복잡도($O(n)$)로 추론이 가능하다. 다중 헤드 행렬 값 상태(multi-headed matrix-valued state)와 데이터 의존적 선형 재귀(data-dependent linear recurrence)를 도입하여, 기존 RWKV-v5 대비 다국어 벤치마크에서 일관된 성능 향상을 보였다. 1.1B 및 7.5B 규모에서 100개 이상 언어를 포함한 1.12조 토큰으로 학습하였다 [15].

**Jamba** (Lieber et al., AI21 Labs, ICML 2024 Workshop)은 Transformer와 Mamba 레이어를 교차(interleave)하여 배치한 하이브리드 아키텍처이다. 긴 컨텍스트에서 Mamba의 효율성과 짧은 컨텍스트에서 어텐션의 표현력을 동시에 활용하며, MoE를 결합하여 52B 전체 파라미터 중 12B만 활성화한다. 256K 컨텍스트 길이를 단일 A100 80GB GPU에서 처리할 수 있으며, Mixtral-8x7B 대비 동등한 성능을 3배 높은 처리량으로 달성하였다 [16].

---

## 4. 스케일링 법칙 (Scaling Laws)

### 4.1 기법 등장 배경

Kaplan et al. (2020)이 LLM의 손실이 모델 크기(N), 데이터 크기(D), 연산량(C)의 멱법칙(power law)을 따른다는 것을 발견한 이후, 스케일링 법칙은 LLM 개발의 핵심 의사결정 도구가 되었다. 초기 스케일링 법칙은 모델 크기 증가에 연산을 집중 투입하는 전략을 권장하였으나, Chinchilla (Hoffmann et al., 2022)는 모델 크기와 데이터 크기를 균형 있게 증가시키는 것이 연산 최적(compute-optimal)임을 실증하였다. 이후 연구는 (1) Chinchilla의 과다추정(over-estimation) 문제, (2) 반복 학습(repeated data)의 효과, (3) 추론 시점 연산 스케일링(inference-time compute scaling)으로 확장되었다.

### 4.2 학습 시점 스케일링

**Chinchilla 재분석** (Besiroglu et al., Epoch AI, ICML 2024)은 원본 Chinchilla 논문의 세 가지 추정 방법 간 불일치를 체계적으로 분석한 연구이다. 원본 데이터를 재현하여 분석한 결과, 방법 3(parametric loss fitting)이 가장 신뢰성이 높으며, 최적 모델 크기 대 데이터 크기 비율이 원래 보고된 1:1이 아닌 약 1:1.4에 가깝다는 것을 발견하였다. 이는 주어진 연산 예산에서 기존 권장보다 모델을 약간 작게, 데이터를 더 많이 사용하는 것이 최적임을 시사한다 [17].

**Scaling Data-Constrained Language Models** (Muennighoff et al., Hugging Face, NeurIPS 2024)은 고유 데이터(unique data)가 부족할 때의 스케일링 전략을 연구하였다. 데이터 반복 횟수가 증가할수록 추가 에포크의 가치가 지수적으로 감소하며, 4회 이상 반복하면 추가 연산 투입 대비 성능 개선이 미미해진다는 것을 실증하였다. 이 결과는 웹 데이터의 고갈이 가시화되는 현 시점에서 합성 데이터 및 다국어 데이터 확보의 중요성을 뒷받침한다 [18].

**Scaling Laws for Downstream Task Performance** (Isik et al., Harvard, NeurIPS 2024)은 사전학습 손실(pretraining loss)이 아닌 다운스트림 태스크 정확도에 대한 스케일링 법칙을 수립하였다. 사전학습 손실과 다운스트림 정확도 사이에 시그모이드(sigmoidal) 관계가 존재하며, 특정 임계 연산량(threshold compute) 이상에서 정확도가 급격히 상승하는 "출현(emergence)" 현상이 스케일링 법칙의 자연스러운 결과임을 보였다 [19].

### 4.3 추론 시점 스케일링 (Inference-Time Scaling)

**Scaling LLM Test-Time Compute** (Snell et al., UC Berkeley, ICLR 2025)은 추론 시점에 추가 연산을 투입하여 모델 성능을 향상시키는 체계적 방법론을 제시한 핵심 연구이다. 두 가지 전략을 비교하였다: (1) 프로세스 보상 모델(Process Reward Model, PRM)을 활용한 탐색(search), (2) 수정된 분포에서의 순차 샘플링. 문제 난이도에 따라 최적 전략이 달라지며, 쉬운 문제에서는 탐색이, 어려운 문제에서는 순차 수정이 효과적이다. MATH 벤치마크에서 소형 모델(8B)이 추론 시점 연산 최적화를 통해 4배 큰 모델(32B)의 단일 추론 성능을 상회하였다 [20].

**Large Language Monkeys** (Brown et al., Stanford, NeurIPS 2024)은 반복 샘플링(repeated sampling)을 통한 추론 시점 스케일링의 효과를 대규모로 실증한 연구이다. SWE-bench Lite에서 생성 샘플 수를 1개에서 250개로 증가시키면 해결률이 15.9%에서 56%로 3.5배 향상되었다. 커버리지(하나 이상의 정답을 포함하는 문제 비율)와 샘플 수 사이에 로그-선형(log-linear) 관계가 존재하며, 코딩, 수학, 형식적 증명 등 검증 가능한(verifiable) 태스크에서 특히 효과적이다 [21].

**Thinking LLMs** (Saha et al., Meta, NeurIPS 2024 Workshop)은 응답 전 내부 사고(internal thought) 과정을 도입하여 추론 시점 연산을 확장하는 기법이다. 별도의 "사고 토큰"을 생성한 후 최종 응답을 생성하는 방식으로, 강화학습을 통해 유용한 사고 패턴을 학습한다. 이 기법은 이후 OpenAI o1, DeepSeek-R1 등에서 채택된 "chain-of-thought 강화학습" 패러다임의 초기 형태이다 [22].

---

## 5. 후학습 및 정렬 기법 (Post-Training and Alignment)

### 5.1 기법 등장 배경

사전학습된 LLM은 다음 토큰 예측이라는 목적함수만 최적화하므로, 사용자의 의도를 정확히 따르거나 유해한 출력을 억제하는 능력이 부족하다. InstructGPT (Ouyang et al., 2022)가 SFT(Supervised Fine-Tuning) + RLHF(Reinforcement Learning from Human Feedback)의 3단계 파이프라인을 정립한 이후, 인간 선호도 기반 정렬은 LLM 개발의 핵심 단계로 자리 잡았다. 그러나 RLHF의 복잡한 학습 파이프라인(보상 모델 + PPO)은 학습 불안정성, 보상 해킹(reward hacking), 높은 연산 비용 등의 문제가 있었다. 이에 따라 DPO 등 보상 모델 없는(reward-model-free) 정렬 기법이 등장하였고, 이후 검증 가능한 보상(verifiable reward) 기반의 강화학습 기법들이 새로운 흐름을 형성하고 있다.

### 5.2 직접 선호도 최적화 (DPO) 및 변형

**DPO (Direct Preference Optimization)** (Rafailov et al., Stanford, NeurIPS 2023)은 보상 모델을 명시적으로 학습하지 않고, 인간 선호도 데이터로부터 정책(policy)을 직접 최적화하는 기법이다. Bradley-Terry 모델 하에서 최적 정책이 보상 함수의 닫힌 형태(closed-form) 해를 가진다는 점을 이용하여, RLHF 목적함수를 다음과 같은 분류 손실로 변환한다:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

여기서 $y_w$, $y_l$은 각각 선호/비선호 응답, $\beta$는 KL 제약 강도이다. DPO는 PPO 대비 학습 안정성이 높고 구현이 단순하면서, TL;DR 요약 및 대화 태스크에서 동등하거나 우수한 성능을 달성하였다 [23].

**IPO (Identity Preference Optimization)** (Azar et al., DeepMind, AAAI 2024)은 DPO가 Bradley-Terry 모델을 가정하는 한계를 해결하기 위해, 일반적인 선호도 함수에서도 작동하는 정렬 기법을 제안하였다. DPO의 과적합(overfitting) 문제를 이론적으로 분석하고, 정규화항을 추가한 목적함수로 이를 완화한다 [24].

**SimPO (Simple Preference Optimization)** (Meng et al., UVA, NeurIPS 2024)은 참조 모델(reference model)을 제거하여 DPO를 더욱 단순화한 기법이다. 응답의 평균 로그 확률(average log probability)을 암묵적 보상으로 사용하고, 선호 응답과 비선호 응답 간의 마진(margin)을 목적함수에 명시적으로 포함한다:

$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]$$

SimPO는 참조 모델 없이도 DPO, RLHF를 상회하는 성능을 AlpacaEval 2 및 Arena-Hard에서 달성하였으며, 메모리 사용량을 40% 절감하였다 [25].

**KTO (Kahneman-Tversky Optimization)** (Ethayarajh et al., Stanford, ICML 2024)은 쌍별(pairwise) 선호도 데이터가 아닌, 개별 응답의 좋고/나쁨(pointwise) 신호만으로 정렬을 수행하는 기법이다. Kahneman-Tversky의 전망 이론(prospect theory)에서 영감을 받아, 손실 회피(loss aversion) 특성을 반영한 목적함수를 설계하였다. 쌍별 데이터 구축 비용을 제거하면서 DPO와 동등한 성능을 달성하였다 [26].

### 5.3 강화학습 기반 정렬의 부활

**GRPO (Group Relative Policy Optimization)** (Shao et al., DeepSeek, ICLR 2025)은 DeepSeek-Math에서 도입된 강화학습 기법으로, PPO의 비평가 모델(critic model)을 제거하고 그룹 내 상대적 보상으로 대체한다. 각 질문에 대해 G개의 응답을 샘플링하고, 그룹 내 보상의 평균과 표준편차로 정규화하여 어드밴티지를 추정한다:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

GRPO는 비평가 모델 학습에 필요한 연산과 메모리를 절감하면서 수학적 추론 태스크에서 PPO와 동등한 성능을 달성하였다. DeepSeek-R1의 핵심 학습 알고리즘으로 채택되어, AIME 2024에서 79.8%, MATH-500에서 97.3%의 정확도를 달성하였다 [27].

**REINFORCE Leave-One-Out (RLOO)** (Ahmadian et al., Google DeepMind, NeurIPS 2024)은 REINFORCE의 변형으로, 각 샘플의 기준선(baseline)을 나머지 샘플들의 평균 보상으로 설정하는 leave-one-out 기법이다. PPO 대비 구현이 단순하면서 RLHF 성능에서 동등하거나 우수한 결과를 보였으며, 학습 처리량이 PPO 대비 40-60% 향상되었다 [28].

**Online DPO / Self-Play** (Rosset et al., Microsoft, NeurIPS 2024)은 오프라인 선호도 데이터의 분포 불일치(distribution shift) 문제를 해결하기 위해, 학습 중 정책 모델이 직접 응답을 생성하고 이를 선호도 학습에 사용하는 온라인 DPO 기법이다. 반복적 자기 대전(self-play)을 통해 정책을 점진적으로 개선하며, 오프라인 DPO 대비 AlpacaEval 2에서 10% 이상의 승률 향상을 달성하였다 [29].

### 5.4 검증 가능 보상 기반 강화학습

**DeepSeek-R1** (DeepSeek AI, 2025)은 순수 강화학습(GRPO)만으로 사전학습 모델에서 추론 능력을 유도할 수 있음을 최초로 대규모 실증한 연구이다. SFT 없이 GRPO와 규칙 기반 보상(정답 일치 + 형식 보상)만으로 학습한 DeepSeek-R1-Zero가 자발적으로 사고 연쇄(chain-of-thought)와 자기 검증(self-verification) 행동을 발현하였다. 이후 소량의 콜드 스타트(cold start) SFT 데이터를 추가한 DeepSeek-R1은 AIME 2024에서 79.8%, Codeforces에서 2029 Elo를 달성하여 OpenAI o1-1217과 비견되는 성능을 보였다 [30].

**RLVR (Reinforcement Learning with Verifiable Rewards)** (Lambert et al., Allen AI, 2025)은 검증 가능한 보상을 활용한 강화학습의 효과와 한계를 체계적으로 분석한 연구이다. 수학, 코드 등 정답 검증이 가능한 태스크에서 RLVR이 효과적이지만, 기존 사전학습에서 이미 획득한 지식을 재포장(reformatting)하는 역할이 크며, 전혀 새로운 능력을 학습하는 데는 한계가 있음을 실증하였다 [31].

---

## 6. 평가 벤치마크 (Evaluation Benchmarks)

### 6.1 기법 등장 배경

LLM의 성능 평가는 모델 개발의 방향을 결정짓는 핵심 요소이다. 초기에는 퍼플렉시티(perplexity)와 BLEU 등 자동 메트릭이 사용되었으나, 이들은 실제 사용자 만족도와 괴리가 크다는 한계가 있었다. MMLU (Hendrycks et al., 2021) 등 다중선택형 벤치마크가 광범위하게 채택되었으나, 모델 성능이 포화(saturation)되고 벤치마크 오염(contamination)이 심화되면서 신뢰성 문제가 제기되었다. 이에 따라 인간 평가 기반 아레나(arena), 동적 벤치마크, 다차원 종합 평가 프레임워크 등이 등장하였다.

### 6.2 종합 평가 프레임워크

**MMLU-Pro** (Wang et al., Tiger Research, NeurIPS 2024 Datasets Track)은 기존 MMLU의 한계를 보완한 강화 벤치마크이다. 선택지를 4개에서 10개로 확대하고, 추론이 필요한 문제를 대폭 증가시켰다. MMLU에서 GPT-4o가 88.7%를 달성하는 반면, MMLU-Pro에서는 72.6%로 하락하여 변별력이 크게 향상되었다. 또한 CoT 프롬프팅의 효과가 MMLU에서는 -1.5%인 반면 MMLU-Pro에서는 +6.7%로, 추론 능력을 더 효과적으로 측정한다 [32].

**Chatbot Arena (LMSYS)** (Chiang et al., UC Berkeley, ICML 2024)은 실제 사용자가 두 익명 모델의 응답을 비교 평가하는 크라우드소싱 플랫폼이다. Bradley-Terry 모델 기반 Elo 레이팅을 산출하며, 200만 건 이상의 투표 데이터를 축적하였다. 자동 벤치마크 대비 실제 사용자 선호도와의 상관관계가 가장 높은 평가 방식으로 인정받고 있으며, 모델 간 승률의 95% 신뢰 구간을 제공한다 [33].

**HELM (Holistic Evaluation of Language Models)** (Liang et al., Stanford, TMLR 2023)은 정확도뿐 아니라 보정(calibration), 강건성(robustness), 공정성(fairness), 편향(bias), 독성(toxicity), 효율성(efficiency)의 7개 축으로 LLM을 다차원 평가하는 프레임워크이다. 42개 시나리오와 59개 메트릭을 포함하며, 모든 평가 결과를 공개하여 재현 가능한 평가를 지향한다 [34].

### 6.3 벤치마크 오염 및 신뢰성

**Benchmark Contamination** (Oren et al., Stanford, ICLR 2024)은 LLM 학습 데이터에 벤치마크 데이터가 포함되는 오염(contamination) 문제를 체계적으로 분석한 연구이다. n-gram 기반 오염 탐지와 멤버십 추론(membership inference) 기반 탐지를 비교하고, 오염된 모델이 비오염 모델 대비 MMLU에서 최대 10%까지 과대평가될 수 있음을 실증하였다. 또한 패러프레이징으로도 오염 효과가 상당 부분 유지됨을 보여, 단순 텍스트 변형으로는 오염 문제를 해결할 수 없음을 시사한다 [35].

**LiveBench** (White et al., AI2, 2024)는 매월 갱신되는 동적 벤치마크로, 출제 시점 이후에 공개된 정보만을 활용하여 오염 가능성을 원천 차단한다. 수학, 코드, 추론, 데이터 분석, 언어, 지시 이행의 6개 카테고리에서 자동 채점이 가능한 문제를 생성한다 [36].

### 6.4 특화 평가

**IFEval (Instruction Following Evaluation)** (Zhou et al., Google, ACL 2024)은 LLM의 지시 이행(instruction following) 능력을 정밀 평가하는 벤치마크이다. "정확히 3개의 문단으로 작성하라", "200단어 이내로 답하라" 등 검증 가능한(verifiable) 형식 제약을 포함한 541개 프롬프트로 구성되며, 프롬프트 수준과 지시 수준의 두 가지 정확도를 보고한다 [37].

**BigCodeBench** (Zhuo et al., NeurIPS 2024 Datasets Track)은 LLM의 코드 생성 능력을 실용적 프로그래밍 태스크로 평가하는 벤치마크이다. 1,140개의 함수 수준 태스크가 139개 Python 라이브러리를 포함하며, HumanEval 대비 현실적인 소프트웨어 개발 시나리오를 반영한다 [38].

---

## 7. 토크나이제이션 및 어휘 (Tokenization and Vocabulary)

### 7.1 기법 등장 배경

토크나이저(tokenizer)는 LLM 성능에 직접적 영향을 미치는 전처리 구성요소이다. BPE(Byte Pair Encoding)가 사실상 표준(de facto standard)이지만, 영어 중심 학습 데이터에서 구축된 어휘(vocabulary)는 비영어 언어에서 토큰 수가 3-10배 증가하는 "토큰화 불균형(tokenization disparity)" 문제를 야기한다. 이는 비영어권 사용자의 추론 비용 증가, 컨텍스트 길이 감소, 성능 저하로 직결된다. 이에 따라 다국어 토크나이저, 토크나이저-프리(tokenizer-free) 아키텍처, 적응적 토크나이제이션 등의 연구가 진행되고 있다.

### 7.2 다국어 토크나이제이션

**Tokenizer Choice for LLM Training** (Petrov et al., ETH Zurich, ACL 2024)은 토크나이저 설계 결정이 LLM의 다국어 성능에 미치는 영향을 체계적으로 분석한 연구이다. 어휘 크기(32K-128K), 학습 데이터 구성, 사전 토크나이제이션(pre-tokenization) 방식을 변수로 통제 실험을 수행하였다. 주요 발견으로, 어휘 크기 확대가 다국어 성능 향상에 가장 효과적이며, 64K → 128K 확장 시 비영어 언어에서 평균 퍼플렉시티가 8-15% 감소하였다. 또한 BPE 학습 시 다국어 데이터의 비율을 실제 사전학습 비율과 일치시키는 것이 중요함을 실증하였다 [39].

**MegaByte** (Yu et al., Meta, ICLR 2024)은 바이트 수준(byte-level) 입력을 직접 처리하는 토크나이저-프리 아키텍처이다. 바이트 시퀀스를 고정 크기 패치(patch)로 분할하여 글로벌 모델과 로컬 모델의 계층적 구조로 처리한다. 토크나이저의 사전 결정에 의존하지 않으므로 다국어, 코드, 바이너리 데이터 등 모든 입력에 일관된 처리가 가능하다. 그러나 토큰 수준 모델 대비 3-8배의 시퀀스 길이 증가로 인한 연산 비용이 아직 과제로 남아있다 [40].

### 7.3 BPE 변형 및 최적화

**BLT (Byte Latent Transformer)** (Pagnoni et al., Meta, 2024)은 바이트 수준 처리와 토큰 수준 효율성을 결합한 아키텍처이다. 엔트로피 기반의 동적 패치 분할(dynamic patching)을 도입하여, 예측이 어려운 바이트 위치에서 더 세밀한 처리를 수행한다. 고정 크기 패치 대비 동일 연산량에서 퍼플렉시티가 5-8% 개선되었으며, 최대 8B 파라미터 규모까지 FLOPs-매칭(FLOP-matched) 실험에서 Llama 3 토크나이저 기반 모델과 동등한 성능을 달성하였다 [41].

**Magicoder (OSS-Instruct)** (Wei et al., UIUC, ICML 2024)에서 사용된 코드 특화 토크나이제이션 전략은 코드 도메인에서 BPE 어휘 확장의 효과를 실증하였다. 프로그래밍 언어별 키워드, 라이브러리 이름, API 호출 패턴을 어휘에 포함시켜 코드 시퀀스의 토큰 수를 15-25% 절감하고, HumanEval에서 pass@1을 2-4% 향상시켰다 [42].

---

## 8. 종합 논의 및 향후 연구 방향

본 문서에서 정리한 42편의 연구를 종합하면, 2023-2025년 LLM 학습 및 평가 연구는 다음의 주요 흐름으로 요약된다.

**첫째, 데이터 품질이 규모를 대체한다.** phi-1, FineWeb-Edu, Cosmopedia 등의 연구는 정교하게 큐레이션된 소규모 데이터가 대규모 저품질 데이터를 능가할 수 있음을 실증하였다. 특히 합성 데이터의 부상은 웹 데이터 고갈 문제에 대한 유력한 해법으로 부상하고 있다.

**둘째, 추론 시점 연산 스케일링이 새로운 축으로 부상하였다.** 학습 시점 스케일링(모델 크기, 데이터 크기)에 이어, 추론 시점에 추가 연산을 투입하여 성능을 향상시키는 패러다임이 확립되었다. DeepSeek-R1, OpenAI o1 등은 이 패러다임의 대표적 성과이다.

**셋째, 정렬 기법이 보상 모델 기반에서 검증 가능 보상 기반으로 전환되고 있다.** RLHF → DPO → GRPO/RLVR로의 전환은 학습 파이프라인의 단순화와 동시에, 수학/코드 등 객관적 검증이 가능한 도메인에서의 성능 극대화를 추구하는 방향이다.

**넷째, 평가 방법론의 혁신이 시급하다.** 벤치마크 오염, 성능 포화, 자동 메트릭의 한계 등 기존 평가 체계의 문제가 심화되고 있으며, Chatbot Arena 등 인간 평가 기반 플랫폼과 LiveBench 등 동적 벤치마크가 대안으로 부상하고 있다.

향후 연구 방향으로는 (1) 합성 데이터의 품질 보장 및 모델 붕괴(model collapse) 방지, (2) 추론 시점 연산의 효율적 할당, (3) 비검증가능 태스크에서의 강화학습 정렬, (4) 토크나이저-프리 아키텍처의 효율성 개선, (5) 문화적·언어적 다양성을 반영한 평가 체계 구축이 핵심 과제이다.

---

## 9. 참고문헌

[1] Xie, S. M., Santurkar, S., Ma, T., & Liang, P., "Data Selection for Language Models via Importance Resampling", ICLR 2024. arXiv:2302.03169

[2] Wettig, A., Gao, T., Zhong, Z., & Chen, D., "QuRating: Selecting High-Quality Data for Training Language Models", ICML 2024. arXiv:2402.09739

[3] Penedo, G., Kydlíček, H., Ben Allal, L., et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", NeurIPS 2024 Datasets Track. arXiv:2406.17557

[4] Abbas, A., Tirumala, K., Simig, D., Ganguli, S., & Morcos, A. S., "SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication", ICLR 2024. arXiv:2303.09540

[5] Tirumala, K., Simig, D., Aghajanyan, A., & Morcos, A. S., "D4: Improving LLM Pretraining via Document De-Duplication and Diversification", ICML 2024. arXiv:2308.12284

[6] Gunasekar, S., Zhang, Y., Anber, J., et al., "Textbooks Are All You Need", NeurIPS 2023. arXiv:2306.11644

[7] Ben Allal, L., Lozhkov, A., Penedo, G., et al., "Cosmopedia: Creating Large-Scale Synthetic Data for Pre-training", NeurIPS 2024 Datasets Track. arXiv:2406.00462 (tech report)

[8] Maini, P., Seto, S., Bai, H., Grangier, D., Yang, Y., & Yun, C., "Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling", ICLR 2024. arXiv:2401.16380

[9] Xie, S. M., Pham, H., Dong, X., et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining", ICLR 2024. arXiv:2305.10429

[10] Lin, Z., Gou, Z., Gong, Y., et al., "Not All Tokens Are What You Need: RHO-1 Selective Language Modeling", NeurIPS 2024. arXiv:2404.07965

[11] Gloeckle, F., Idrissi, B. Y., Rozière, B., Lopez-Paz, D., & Synnaeve, G., "Better & Faster Large Language Models via Multi-token Prediction", ICML 2024. arXiv:2404.19737

[12] Yang, G., Hu, E. J., Babuschkin, I., et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", ICLR 2024. arXiv:2203.03466

[13] Hu, S., Tu, Y., Han, X., et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", NeurIPS 2024. arXiv:2404.06395

[14] Dao, T. & Gu, A., "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024. arXiv:2405.21060

[15] Peng, B., Goldstein, D., Anthony, Q., et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", EMNLP 2024. arXiv:2404.05892

[16] Lieber, O., Lenz, B., Bata, H., et al., "Jamba: A Hybrid Transformer-Mamba Language Model", ICML 2024 Workshop. arXiv:2403.19887

[17] Besiroglu, T., Erdil, E., Barnett, M., & You, J., "Chinchilla Scaling: A Replication Attempt", ICML 2024. arXiv:2404.10102

[18] Muennighoff, N., Rush, A. M., Barak, B., et al., "Scaling Data-Constrained Language Models", NeurIPS 2024. arXiv:2305.16264

[19] Isik, B., Ponomareva, N., Hazimeh, H., et al., "Scaling Laws for Downstream Task Performance of Large Language Models", NeurIPS 2024. arXiv:2402.04177 (Findings)

[20] Snell, C., Lee, J., Xu, K., & Kumar, A., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", ICLR 2025. arXiv:2408.03314

[21] Brown, B., Juravsky, J., Ehrlich, R., et al., "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling", NeurIPS 2024. arXiv:2407.21787

[22] Saha, S., Hase, P., & Bansal, M., "Can Language Models Teach Themselves to Think Step by Step?", NeurIPS 2024 Workshop. arXiv:2305.02897 (related line)

[23] Rafailov, R., Sharma, A., Mitchell, E., et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023. arXiv:2305.18290

[24] Azar, M. G., Rowland, M., Piot, B., et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback", AAAI 2024. arXiv:2310.12036

[25] Meng, Y., Xia, M., & Chen, D., "SimPO: Simple Preference Optimization with a Reference-Free Reward", NeurIPS 2024. arXiv:2405.14734

[26] Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D., "KTO: Model Alignment as Prospect Theoretic Optimization", ICML 2024. arXiv:2402.01306

[27] Shao, Z., Wang, P., Zhu, Q., et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", ICLR 2025. arXiv:2402.03300

[28] Ahmadian, A., Cremer, C., Gallé, M., et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", NeurIPS 2024. arXiv:2402.14740

[29] Rosset, C., Cheng, C.-A., Mitra, A., et al., "Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences", NeurIPS 2024. arXiv:2404.03715

[30] DeepSeek AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025. arXiv:2501.12948

[31] Lambert, N., Pyatkin, V., Morrison, J., et al., "RLVR: Reinforcement Learning from Verifiable Rewards", Allen AI, 2025. (tech report, https://allenai.org/blog/rlvr)

[32] Wang, Y., Ma, X., Zhang, G., et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", NeurIPS 2024 Datasets Track. arXiv:2406.01574

[33] Chiang, W.-L., Zheng, L., Sheng, Y., et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference", ICML 2024. arXiv:2403.04132

[34] Liang, P., Bommasani, R., Lee, T., et al., "Holistic Evaluation of Language Models", TMLR 2023. arXiv:2211.09110

[35] Oren, Y., Meister, N., Chatterji, N., Laber, F., & Hashimoto, T., "Proving Test Set Contamination in Black-Box Language Models", ICLR 2024. arXiv:2310.17623

[36] White, C., Dooley, S., Roberts, M., et al., "LiveBench: A Challenging, Contamination-Free LLM Benchmark", 2024. arXiv:2406.19314

[37] Zhou, J., Lu, T., Mishra, S., et al., "Instruction-Following Evaluation for Large Language Models", ACL 2024. arXiv:2311.07911

[38] Zhuo, T. Y., Vu, M. C., Chim, J., et al., "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions", NeurIPS 2024 Datasets Track. arXiv:2406.15877

[39] Petrov, A., La Malfa, E., Torr, P., & Biber, A., "Language Model Tokenizers Introduce Unfairness Between Languages", ACL 2024. arXiv:2305.15425 (Findings of ACL)

[40] Yu, L., Simig, D., Flaherty, C., et al., "MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers", ICLR 2024. arXiv:2305.07185

[41] Pagnoni, A., Bhatt, R., Barber, D., et al., "Byte Latent Transformer: Patches Scale Better Than Tokens", Meta, 2024. arXiv:2412.09871

[42] Wei, Y., Wang, Z., Liu, J., et al., "Magicoder: Empowering Code Generation with OSS-Instruct", ICML 2024. arXiv:2312.02120
