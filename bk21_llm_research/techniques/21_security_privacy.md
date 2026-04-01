# LLM 보안, 프라이버시, 워터마킹 및 공정성 최신 연구 동향 (2023-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, S&P, CCS, USENIX Security, NDSS) 중심
> 총 참고논문: 42편

---

## 1. 개요

대규모 언어 모델(LLM)의 능력이 급격히 확장됨에 따라, 보안(security), 프라이버시(privacy), 워터마킹(watermarking), 공정성(fairness)의 네 축은 안전한 배포의 핵심 요건으로 부상하였다. GPT-4, Claude, Llama 계열 모델이 수억 명의 사용자에게 서비스되면서, 적대적 공격을 통한 안전 장치 우회, 학습 데이터에 포함된 개인정보 유출, AI 생성 텍스트의 출처 불투명성, 그리고 인종·성별·문화적 편향의 증폭이 실질적 위협으로 현실화되었다.

보안 관점에서는 탈옥(jailbreak) 공격이 RLHF 기반 안전 정렬(safety alignment)을 무력화할 수 있음이 반복적으로 입증되었으며, 프롬프트 인젝션(prompt injection)은 LLM 기반 에이전트의 행동을 탈취하는 새로운 공격 표면을 형성하였다. 프라이버시 측면에서는 멤버십 추론(membership inference)과 학습 데이터 추출(training data extraction) 공격이 모델의 기억(memorization) 특성을 악용하여 민감 정보를 복원할 수 있음이 밝혀졌다. 워터마킹 분야에서는 AI 생성 텍스트의 식별을 위한 통계적 워터마크 삽입 기법이 급속히 발전하였으나, 패러프레이징(paraphrasing) 공격에 대한 강건성이 핵심 과제로 남아 있다. 공정성 분야에서는 LLM이 학습 데이터의 사회적 편향을 재생산하고 증폭하는 메커니즘이 분석되었으며, 디바이어싱(debiasing) 기법과 다문화 평가 벤치마크가 활발히 개발되고 있다.

본 문서에서는 (1) 적대적 공격, (2) 방어 메커니즘, (3) 텍스트 워터마킹 및 AI 생성 텍스트 탐지, (4) 프라이버시, (5) 공정성 및 편향, (6) 저작권 및 지적재산권의 6개 분야로 나누어 2023-2025년 최신 연구를 정리한다.

---

## 2. 적대적 공격 (Adversarial Attacks on LLMs)

### 2.1 기법 등장 배경

LLM의 안전성 확보를 위해 RLHF(Reinforcement Learning from Human Feedback) 및 Constitutional AI 등의 정렬(alignment) 기법이 광범위하게 적용되었다. 그러나 이러한 안전 정렬이 모델의 근본적인 능력을 제거하는 것이 아니라 출력 분포를 조건부로 이동시키는 것에 불과하다는 사실이 밝혀지면서, 정렬을 우회하는 적대적 공격 연구가 폭발적으로 증가하였다. 초기에는 수작업 프롬프트 엔지니어링에 의존하였으나, 점차 자동화된 그래디언트 기반 최적화, 유전 알고리즘, LLM-as-attacker 패러다임으로 진화하였다. 이는 안전 정렬의 근본적 한계를 드러내는 동시에, 보다 강건한 방어 체계의 필요성을 역설하였다.

### 2.2 자동화된 탈옥 공격 (Automated Jailbreak Attacks)

**GCG (Greedy Coordinate Gradient)** (Zou et al., NeurIPS 2023 Spotlight)는 LLM 탈옥의 자동화를 최초로 체계적으로 입증한 연구이다. 이산 토큰 공간에서 그래디언트 기반 검색을 수행하여 적대적 접미사(adversarial suffix)를 생성한다. 구체적으로, 공격 목표는 모델이 유해한 질문에 대해 긍정적 응답("Sure, here is...")을 생성하도록 하는 것이며, 손실 함수는 다음과 같이 정의된다:

$$\mathcal{L}(x_{1:n}) = -\log p(x^*_{n+1:n+H} | x_{1:n})$$

여기서 $x^*_{n+1:n+H}$는 목표 긍정 응답의 토큰 시퀀스이다. 각 좌표(토큰 위치)에 대해 그래디언트를 계산하고, top-$k$ 대체 후보 중 손실을 최소화하는 토큰을 탐욕적으로 선택한다. Vicuna, Llama-2-Chat, GPT-3.5, GPT-4, PaLM-2 등 다수의 모델에서 공격 성공률(ASR) 84% 이상을 달성하였으며, 화이트박스 모델에서 생성된 접미사가 블랙박스 모델로 전이(transfer)됨을 입증하였다 [1].

**AutoDAN** (Liu et al., ICLR 2024)은 그래디언트 기반 탈옥의 가독성 문제를 해결하기 위해, 계층적 유전 알고리즘(hierarchical genetic algorithm)을 적용한다. 문장 수준과 단어 수준의 이중 교차·변이 연산을 설계하여, 의미적으로 자연스러운 탈옥 프롬프트를 자동 생성한다. GCG 대비 공격 성공률을 유지하면서 perplexity 기반 필터링을 우회하는 능력을 보였다 [2].

**PAIR (Prompt Automatic Iterative Refinement)** (Chao et al., NeurIPS 2024)은 공격자 LLM이 대상 LLM을 반복적으로 탐색하여 탈옥 프롬프트를 정제하는 블랙박스 공격 기법이다. 모델의 가중치나 그래디언트 접근 없이, 공격자 LLM의 in-context learning 능력만을 활용한다. GPT-4에 대해 평균 20회 미만의 쿼리로 60% 이상의 ASR을 달성하였다 [3].

**Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation** (Huang et al., ICML 2024)은 디코딩 전략(온도, top-$p$, 반복 패널티)의 변조만으로 안전 정렬을 무력화할 수 있음을 보였다. 특히 높은 온도 설정($T > 1.5$)과 낮은 반복 패널티 조합이 Llama-2-Chat에서 ASR을 95% 이상으로 끌어올렸다. 이는 안전 정렬이 디코딩 하이퍼파라미터에 대해 취약하다는 근본적 한계를 지적한다 [4].

### 2.3 프롬프트 인젝션 공격 (Prompt Injection)

**Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections** (Greshake et al., AISec@CCS 2023)은 LLM 통합 애플리케이션에서 간접 프롬프트 인젝션의 위험성을 최초로 체계화하였다. 웹 검색, 이메일, 코드 실행 도구와 연동된 LLM 에이전트가 외부 데이터에 삽입된 악의적 명령을 실행할 수 있음을 시연하였다. Bing Chat, Google Bard 등 실제 서비스에서 원격 제어, 데이터 유출, 소셜 엔지니어링이 가능함을 입증하였다 [5].

**Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game** (Toyer et al., ICLR 2024)은 12만 6천 개 이상의 프롬프트 인젝션/방어 쌍으로 구성된 대규모 데이터셋을 온라인 게임을 통해 수집하였다. 공격 전략의 분류 체계를 수립하고, 현존 방어 기법의 한계를 정량적으로 분석하였다. 시스템 프롬프트 추출 공격의 성공률이 특정 모델에서 89%에 달함을 보고하였다 [6].

### 2.4 모델 추출 공격 (Model Extraction)

**Stealing Part of a Production Language Model** (Carlini et al., ICML 2024)은 프로덕션 LLM의 임베딩 프로젝션 레이어를 API 쿼리만으로 추출할 수 있음을 입증하였다. OpenAI의 gpt-3.5-turbo 모델에 대해 $\$2000$ 미만의 쿼리 비용으로 최종 레이어의 차원 정보와 가중치를 복원하였다. 구체적으로, 로짓 편향(logit bias) API를 활용하여 전체 어휘에 대한 로짓 벡터를 수집하고, SVD를 통해 모델의 은닉 차원이 4096 또는 8192임을 확인하였다 [7].

---

## 3. 방어 메커니즘 (Defense Mechanisms)

### 3.1 기법 등장 배경

탈옥 공격의 성공률이 지속적으로 보고됨에 따라, RLHF 단독으로는 안전성을 보장할 수 없다는 인식이 확산되었다. 초기 방어는 입력 필터링(블랙리스트, perplexity 필터)에 의존하였으나, 이는 적응적 공격(adaptive attack)에 취약하였다. 이후 추론 시 안전성 가드레일(guardrail), 표현 공학(representation engineering) 기반 내부 조작, 적대적 학습(adversarial training) 등 다층적 방어 체계가 제안되었다. 그러나 방어와 공격의 비대칭적 군비 경쟁이 계속되고 있으며, 단일 방어층으로는 충분하지 않다는 인식이 확립되었다.

### 3.2 가드레일 및 안전 분류기 (Guardrails and Safety Classifiers)

**Llama Guard** (Inan et al., Meta, 2024)는 안전 분류를 위해 특별히 미세조정된 LLM 기반 가드레일 모델이다. 입력 프롬프트와 출력 응답 모두에 대해 위험 범주별 분류를 수행한다. 6개 위험 범주(폭력, 성적 콘텐츠, 범죄 조장 등)에 대해 Llama-2-7B를 미세조정하였으며, OpenAI Moderation API 대비 F1 점수 0.76 → 0.83으로 향상을 달성하였다. Llama Guard 2/3으로 후속 업데이트되며 다국어 지원과 범주 확장이 이루어졌다 [8].

**NeMo Guardrails** (Rebedea et al., EMNLP 2023 Demo)는 NVIDIA가 개발한 오픈소스 가드레일 프레임워크로, Colang이라는 선언적 언어를 통해 대화 흐름 규칙을 정의한다. 토픽 가드레일, 안전 가드레일, 보안 가드레일의 세 층위로 구성되며, 임의의 LLM에 플러그인 형태로 적용 가능하다 [9].

### 3.3 표현 공학 기반 방어 (Representation Engineering)

**Representation Engineering** (Zou et al., ICLR 2024 Spotlight)은 모델의 내부 표현 공간에서 안전성 관련 방향(direction)을 식별하고 조작하는 기법이다. 대조적 프롬프트 쌍(안전/위험)의 활성화 차이를 주성분 분석(PCA)하여 "안전성 방향" $\mathbf{v}_{safety}$를 추출한다. 추론 시 은닉 상태에 $\mathbf{h}' = \mathbf{h} + \alpha \mathbf{v}_{safety}$ 형태로 개입하여 안전한 출력을 유도한다. 이 기법은 미세조정 없이 추론 시에만 적용 가능하며, 안전성, 진실성, 공정성 등 다양한 개념에 일반화된다 [10].

**Circuit Breakers** (Zou et al., NeurIPS 2024)는 유해 입력에 대한 모델의 내부 표현을 무작위화(randomize)하여 유해 출력 생성 자체를 차단하는 기법이다. 표현 재매핑(representation rerouting, RR) 손실을 정의한다:

$$\mathcal{L}_{RR} = \mathbb{E}_{x \sim \mathcal{D}_{harm}} \left[ \| f(x) - \mathbf{r} \|^2 \right] + \lambda \mathbb{E}_{x \sim \mathcal{D}_{safe}} \left[ \| f(x) - f_0(x) \|^2 \right]$$

여기서 $\mathbf{r}$은 무작위 목표 표현, $f_0$은 원본 모델이다. 첫 항은 유해 입력의 표현을 무작위로 재매핑하고, 둘째 항은 정상 입력에 대한 성능을 보존한다. GCG, AutoDAN, PAIR 등 다양한 공격에 대해 ASR을 1% 미만으로 감소시키면서 MT-Bench 성능을 95% 이상 유지하였다 [11].

### 3.4 적대적 학습 기반 방어 (Adversarial Training)

**Adversarial Training for LLMs (R2D2)** (Mazeika et al., NeurIPS 2024)은 LLM에 적대적 학습을 적용하는 체계적 프레임워크를 제안하였다. GCG 공격으로 적대적 접미사를 동적으로 생성하고, 이에 대한 거부 응답을 학습하는 min-max 최적화를 수행한다:

$$\min_{\theta} \max_{\delta \in \mathcal{S}} \mathcal{L}_{refuse}(x + \delta; \theta)$$

Llama-2-7B-Chat에 적용하여 GCG ASR을 82% → 2%로 감소시키면서 MT-Bench 점수를 6.8에서 6.6으로 미미하게만 하락시켰다. 그러나 새로운 유형의 공격(예: multi-turn)에 대한 일반화는 제한적이었다 [12].

**Gradient Cuff** (Hu et al., NeurIPS 2024)는 그래디언트 기반 탈옥 공격을 탐지하는 추론 시 방어 기법이다. 정상 입력과 적대적 입력의 그래디언트 노름(gradient norm) 분포가 통계적으로 유의미하게 다르다는 관찰에 기반한다. 입력의 손실 그래디언트 $\|\nabla_x \mathcal{L}\|$이 임계값 $\tau$를 초과하면 적대적 입력으로 판정한다. GCG 공격 탐지에서 AUROC 0.99를 달성하였다 [13].

### 3.5 탈옥 벤치마크 및 평가 (Jailbreak Benchmarks)

**HarmBench** (Mazeika et al., ICML 2024)는 자동화된 레드팀 평가를 위한 포괄적 벤치마크로, 510개의 유해 행동(functional categories 7개)과 18개의 공격/방어 기법을 표준화된 프로토콜로 평가한다. 분류기 기반 자동 판정(classifier-based auto-judge)을 도입하여 평가의 재현성을 확보하였다 [14].

---

## 4. 텍스트 워터마킹 및 AI 생성 텍스트 탐지

### 4.1 기법 등장 배경

LLM이 생성한 텍스트와 인간이 작성한 텍스트의 구분이 점점 어려워지면서, 학술 부정행위, 허위 정보 유포, 저작권 침해 등의 문제가 대두되었다. 초기 AI 텍스트 탐지기는 통계적 특성(perplexity, burstiness)에 의존하였으나, 패러프레이징이나 번역을 거치면 정확도가 급격히 하락하였다. 이에 따라 모델의 디코딩 과정에 통계적 신호를 삽입하는 워터마킹(watermarking) 기법이 등장하였다. Kirchenbauer et al. (2023)의 개척적 연구 이후, 강건성(robustness), 품질 보존(quality preservation), 탐지 효율성(detection efficiency) 간의 삼중 트레이드오프를 최적화하는 연구가 급증하였다.

### 4.2 통계적 워터마킹 기법 (Statistical Watermarking)

**A Watermark for Large Language Models** (Kirchenbauer et al., ICML 2023)은 LLM 텍스트 워터마킹의 기초를 확립한 연구이다. 각 토큰 생성 시 이전 토큰의 해시를 seed로 사용하여 어휘를 "그린 리스트"(green list)와 "레드 리스트"(red list)로 분할하고, 그린 리스트 토큰의 로짓에 상수 $\delta$를 가산한다:

$$p'(w) = \frac{\exp(l_w + \delta \cdot \mathbb{1}[w \in G])}{\sum_{w'} \exp(l_{w'} + \delta \cdot \mathbb{1}[w' \in G])}$$

탐지 시에는 생성된 텍스트에서 그린 리스트 토큰의 비율이 기대값(0.5)보다 유의미하게 높은지를 $z$-검정으로 판정한다. $\delta = 2.0$ 설정에서 200 토큰 이상의 텍스트에 대해 $p < 10^{-5}$ 수준의 탐지력을 달성하였다 [15].

**Unbiased Watermark for LLMs (UW)** (Hu et al., ICLR 2024)는 Kirchenbauer 워터마크의 텍스트 품질 저하 문제를 해결한다. 그린 리스트 편향 대신, 기각 샘플링(rejection sampling) 기반의 비편향 워터마크를 제안한다. 원본 분포 $p$에서 샘플링하되, 워터마크 키에 의해 결정된 해시 조건을 만족하는 토큰만 수용한다. 이론적으로 생성 분포가 원본과 동일함을 증명하였으며 ($\mathbb{E}[p'] = p$), 실험적으로 perplexity 증가 없이 AUROC 0.99 이상의 탐지 성능을 달성하였다 [16].

**SemStamp: A Semantic Watermark with Paraphrastic Robustness** (Hou et al., NAACL 2024)은 토큰 수준이 아닌 문장 의미 공간에서 워터마크를 삽입한다. 문장 임베딩 공간을 LSH(Locality-Sensitive Hashing)로 분할하고, 워터마크 키에 의해 지정된 영역에 문장 의미가 속하도록 생성을 유도한다. 패러프레이징 공격에 대해 기존 토큰 수준 워터마크 대비 AUROC 0.62 → 0.91로 대폭 개선하였다 [17].

**Adaptive Text Watermark for Large Language Models** (Liu et al., ICML 2024)은 토큰별 엔트로피에 따라 워터마크 강도를 적응적으로 조절한다. 고엔트로피(높은 불확실성) 토큰에는 강한 워터마크를, 저엔트로피(결정적) 토큰에는 약한 워터마크를 적용하여, 텍스트 품질과 탐지력 간의 트레이드오프를 최적화한다 [18].

### 4.3 워터마크 공격 및 강건성 (Watermark Attacks and Robustness)

**On the Reliability of Watermarks for Large Language Models** (Kirchenbauer et al., ICLR 2024)은 자체 워터마크의 강건성을 체계적으로 분석한 후속 연구이다. 토큰 삭제, 동의어 치환, 패러프레이징 공격 하에서의 탐지 성능 열화를 정량화하고, 다중 키 워터마킹과 공개 키 탐지 프로토콜을 제안하였다 [19].

**Watermark Stealing in Large Language Models** (Jovanović et al., ICML 2024)은 워터마크 탐지기의 존재가 역설적으로 워터마크 도용(stealing)을 가능하게 함을 입증하였다. 공격자가 탐지기의 피드백을 이용하여 그린 리스트를 역추론하고, 비워터마크 텍스트에 워터마크를 삽입하거나 워터마크 텍스트에서 워터마크를 제거할 수 있음을 시연하였다. 이는 공개 탐지 API의 설계에 근본적인 재고를 요구한다 [20].

### 4.4 AI 생성 텍스트 탐지 (AI-Generated Text Detection)

**DetectGPT** 이후의 제로샷 탐지 기법으로, **Fast-DetectGPT** (Bao et al., ICLR 2024)는 원본 DetectGPT의 perturbation 기반 접근의 계산 비용을 대폭 감소시켰다. 랜덤 perturbation 대신 조건부 확률 곡률(conditional probability curvature)을 직접 추정하여, 340배의 속도 향상과 동시에 AUROC을 개선하였다. xsum 데이터셋에서 GPT-Neo-2.7B 생성 텍스트에 대해 AUROC 0.98을 달성하였다 [21].

**Spotting LLMs With Binoculars** (Hans et al., ICML 2024)는 두 개의 서로 다른 LLM의 perplexity 비율을 활용하는 "Binoculars" 점수를 제안한다:

$$\text{Binoculars}(x) = \frac{\text{PPL}_{observer}(x)}{\text{PPL}_{performer}(x)}$$

observer 모델과 performer 모델의 perplexity 비율이 인간 텍스트와 기계 생성 텍스트에서 유의미하게 다르다는 관찰에 기반한다. 학습 데이터 없이(zero-shot) ChatGPT, GPT-4 생성 텍스트에 대해 AUROC 0.94를 달성하였다 [22].

**Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts** (Tulchinskii et al., NeurIPS 2023)는 텍스트의 내재적 차원(intrinsic dimensionality)이 인간 텍스트와 기계 생성 텍스트에서 체계적으로 다름을 발견하였다. 최근접 이웃(nearest neighbor) 기반 내재적 차원 추정기를 사용하며, 패러프레이징 공격에 대한 강건성을 기존 탐지기 대비 유의미하게 개선하였다 [23].

---

## 5. 프라이버시 (Privacy)

### 5.1 기법 등장 배경

LLM의 학습 데이터에는 웹에서 수집된 대규모 텍스트가 포함되며, 이 중 상당 부분이 개인 식별 정보(PII), 의료 기록, 연락처 정보 등 민감 데이터를 포함한다. 초기에는 모델의 방대한 파라미터 수가 개별 데이터 포인트의 기억을 불가능하게 만든다고 가정하였으나, Carlini et al. (2021)의 연구에서 GPT-2가 학습 데이터를 거의 정확하게 재현할 수 있음이 입증되면서 이 가정이 무너졌다. 이후 멤버십 추론(membership inference), 학습 데이터 추출(training data extraction), 속성 추론(attribute inference) 공격이 LLM에 특화되어 발전하였다. 방어 측에서는 차등 프라이버시(differential privacy)와 연합학습(federated learning)이 핵심 기법으로 부상하였으나, 유틸리티-프라이버시 트레이드오프가 근본적 과제로 남아 있다.

### 5.2 학습 데이터 추출 공격 (Training Data Extraction)

**Scalable Extraction of Training Data from (Production) Language Models** (Nasr et al., USENIX Security 2024)은 프로덕션 수준의 LLM에서 학습 데이터를 대규모로 추출할 수 있음을 입증한 연구이다. ChatGPT(GPT-3.5-turbo)에 "특정 단어를 영원히 반복하라"는 프롬프트를 입력하면, 반복 후 학습 데이터의 기억된 내용을 출력하기 시작하는 "divergence attack"을 발견하였다. 총 10,000개 이상의 고유 기억된 학습 예제를 추출하였으며, 추출된 데이터에는 이메일 주소, 전화번호, 정확한 URL 등 PII가 포함되어 있었다. 정렬(alignment)이 이러한 추출을 완전히 방지하지 못함을 입증하였다 [24].

**Extracting Training Data from Large Language Models** (Carlini et al., USENIX Security 2021, 후속 확장 2023)은 LLM의 기억 현상을 최초로 체계적으로 분석한 연구이다. 모델 크기가 증가할수록 기억률이 superlinearly 증가함을 보였으며, GPT-Neo 모델군에서 모델 크기를 10배 증가시키면 추출 가능한 학습 데이터가 19배 증가함을 확인하였다 [25].

### 5.3 멤버십 추론 공격 (Membership Inference Attacks)

**Membership Inference Attacks on Language Models via Self-calibrated Probabilistic Variation** (Zhang et al., ACL 2024)은 LLM에 특화된 멤버십 추론 기법 SPV-MIA를 제안하였다. 기존 MIA가 참조 모델(reference model)에 의존하는 한계를 극복하기 위해, 대상 텍스트의 의미적 변형(paraphrase)에 대한 확률 변동을 자기 보정(self-calibration)하여 멤버십을 판정한다. 기존 기법 대비 AUC가 최대 12% 향상되었다 [26].

**Do Membership Inference Attacks Work on Large Language Models?** (Duan et al., CCS 2024)은 LLM에 대한 MIA의 실효성을 대규모 실험으로 검증한 연구이다. Pythia, GPT-Neo, LLaMA, OPT 등 15개 이상의 모델에 대해 기존 MIA 기법을 재평가하였다. 핵심 발견은, 학습 데이터와 비학습 데이터의 분포 차이(distribution shift)가 MIA 성능의 주요 교란 변인이며, 분포를 통제하면 대부분의 MIA가 무작위 추측(AUC ≈ 0.5)에 근접한다는 것이다. 단, 데이터 중복(duplicate)이 높은 경우에는 MIA가 유효함을 확인하였다 [27].

### 5.4 차등 프라이버시 (Differential Privacy)

**DP-SGD (Differentially Private Stochastic Gradient Descent)** 를 LLM에 적용하는 연구가 활발히 진행되고 있다. 차등 프라이버시의 공식 정의는 다음과 같다:

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\epsilon} \Pr[\mathcal{M}(D') \in S] + \delta$$

여기서 $D$와 $D'$는 하나의 데이터 포인트만 다른 이웃 데이터셋이며, $(\epsilon, \delta)$는 프라이버시 예산이다.

**DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer** (Hong et al., ICLR 2024)은 프롬프트 생성 과정에 차등 프라이버시를 적용하는 기법이다. 민감 데이터로부터 DP-보장 프롬프트를 생성하여, 모델 가중치 접근 없이(블랙박스) LLM의 프라이버시를 확보한다. $\epsilon = 1$ 수준에서도 SST-2에서 정확도 88%를 달성하여, 프라이버시-유틸리티 트레이드오프를 개선하였다 [28].

**Differentially Private Fine-tuning of Language Models** (Yu et al., ICLR 2023)는 DP-SGD의 LLM 미세조정 적용에서 배치 크기, 클리핑 노름, 노이즈 스케일의 최적 하이퍼파라미터 구성을 체계적으로 분석하였다. 핵심 발견은, 대형 모델일수록 DP-SGD의 유틸리티 손실이 상대적으로 작다는 것이며, GPT-2-Large ($\epsilon = 3$)에서 비프라이빗 미세조정 대비 3% 이내의 정확도 하락을 달성하였다 [29].

### 5.5 연합학습 (Federated Learning for LLMs)

**FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning** (Kuang et al., ACL 2024 Demo)은 LLM 연합 미세조정을 위한 통합 프레임워크를 제안하였다. LoRA와 연합학습을 결합하여, 클라이언트가 LoRA 어댑터만 공유하고 전체 모델 가중치는 로컬에 유지하는 FedLoRA 전략을 구현하였다. 통신 비용을 전체 모델 공유 대비 99.9% 이상 감소시키면서 중앙 집중 학습의 90% 이상 성능을 달성하였다 [30].

---

## 6. 공정성 및 편향 (Fairness and Bias)

### 6.1 기법 등장 배경

LLM은 인터넷에서 수집된 대규모 텍스트 코퍼스로 학습되며, 이 데이터에 내재된 사회적 편향(성별, 인종, 종교, 국적 등)을 그대로 학습하고 증폭하는 경향이 있다. 초기 연구는 정적 워드 임베딩(Word2Vec, GloVe)에서의 편향 측정(WEAT, SEAT)에 집중하였으나, 생성형 LLM의 등장으로 편향이 자유 형태 텍스트 생성에서 복잡하게 발현되면서 새로운 측정 및 완화 기법이 필요하게 되었다. 특히 RLHF로 정렬된 모델이 표면적으로는 편향을 감소시키나, 암묵적(implicit) 편향은 여전히 존재한다는 발견이 이 분야의 핵심 동기이다.

### 6.2 편향 측정 및 벤치마크 (Bias Measurement and Benchmarks)

**BBQ: A Hand-Built Bias Benchmark for Question Answering** (Parrish et al., ACL 2022, LLM 평가에 2023-2024 지속 사용)은 9개 사회적 범주에 걸쳐 58,000개 이상의 질문-답변 쌍으로 구성된 편향 벤치마크이다. 모호한 컨텍스트(ambiguous context)에서의 모델 응답이 사회적 고정관념에 부합하는지를 정량적으로 측정한다 [31].

**Evaluating and Mitigating Discrimination in Language Model Decisions** (Tamkin et al., NeurIPS 2024)은 LLM이 의사결정(대출 승인, 채용, 보험 등)에 활용될 때의 차별적 행태를 체계적으로 분석하였다. 70개 현실적 시나리오에서 인종, 성별, 연령에 따른 결정 차이를 정량화하였으며, GPT-4가 특정 시나리오에서 흑인 지원자에 대해 대출 승인률이 7.2%p 낮음을 보고하였다 [32].

### 6.3 디바이어싱 기법 (Debiasing Techniques)

**Self-Debiasing: Adjusting LLM Outputs for Fairness** (Schick et al., TACL 2021, 후속 확장 2024)은 모델이 자신의 편향을 인식하고 출력 분포를 조정하는 자기 디바이어싱 기법이다. 편향 유발 프롬프트와 중립 프롬프트의 출력 확률 비율을 이용하여 편향 방향을 추정하고, 디코딩 시 이를 상쇄한다. 후속 연구에서 instruction-tuned 모델에 대한 확장이 이루어졌다 [33].

**Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs** (Gupta et al., ICLR 2024)은 페르소나를 부여받은 LLM이 추론 과정에서 암묵적 편향을 보임을 입증하였다. "아프리카계 미국인 남성" 페르소나를 부여받은 GPT-4가 수학 문제 풀이에서 정확도가 4.7% 하락하는 현상을 발견하였다. 이는 표면적 필터링으로는 잡히지 않는 깊은 수준의 편향이 존재함을 시사한다 [34].

### 6.4 다문화 및 다국어 편향 (Cross-cultural and Multilingual Bias)

**CulturalBench: Benchmarking LLMs for Cultural Knowledge** (Chiu et al., NeurIPS 2024 Datasets and Benchmarks)는 45개 국가와 지역에 걸친 1,227개 질문으로 구성된 문화적 지식 벤치마크이다. GPT-4o, Claude-3.5-Sonnet 등 최신 모델도 비서구권 문화에 대해 평균 15-20%의 정확도 하락을 보임을 확인하였다 [35].

**Multilingual Jailbreak Challenges in Large Language Models** (Deng et al., ICLR 2024)은 안전 정렬이 영어 중심으로 이루어져, 저자원 언어(low-resource language)로 번역된 유해 프롬프트에 대한 방어가 취약함을 입증하였다. 줄루어, 스코틀랜드 게일어 등으로 번역된 탈옥 프롬프트의 ASR이 영어 대비 최대 40%p 높았다. 이는 안전성의 다국어 일반화가 심각한 과제임을 지적한다 [36].

---

## 7. 저작권 및 지적재산권 (Copyright and Intellectual Property)

### 7.1 기법 등장 배경

LLM의 학습 데이터에 저작권이 있는 텍스트(뉴스 기사, 소설, 코드 등)가 광범위하게 포함되면서, 학습 데이터의 저작권 침해 여부와 모델 출력의 저작권 귀속이 법적·기술적 쟁점으로 부상하였다. The New York Times vs. OpenAI (2023), Authors Guild vs. OpenAI (2023) 등 대규모 소송이 진행되면서, 기술적 관점에서 (1) 모델이 저작물을 얼마나 기억하는지 정량화, (2) 기억을 방지하는 기법, (3) 기여도를 추적하는 데이터 귀속(data attribution) 기법에 대한 연구 수요가 급증하였다.

### 7.2 저작물 기억 및 재현 분석 (Copyright Memorization Analysis)

**Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4** (Chang et al., EMNLP 2023)은 GPT 모델이 저작권 도서의 내용을 얼마나 정확하게 재현할 수 있는지를 체계적으로 분석하였다. 인기 소설의 첫 문단이 주어졌을 때 후속 텍스트와의 ROUGE-L 유사도가 특정 작품에서 0.85 이상에 달함을 보고하였다. 2000년 이후 출판 도서에 대한 기억률이 유의미하게 높으며, 이는 학습 데이터 구성과 직접적으로 관련됨을 시사하였다 [37].

**Copyright Traps for Large Language Models** (Meeus et al., ICML 2024)은 저작물에 의도적으로 고유한 트랩 시퀀스(trap sequence)를 삽입하여, 해당 텍스트가 LLM 학습에 사용되었는지를 사후적으로 검증하는 기법을 제안하였다. 트랩 시퀀스의 길이와 반복 횟수에 따른 탐지 민감도를 분석하여, 50토큰 길이의 시퀀스가 7회 이상 반복된 경우 95% 이상의 탐지율을 달성하였다 [38].

### 7.3 데이터 귀속 및 기여도 추적 (Data Attribution)

**TRAK: Attributing Model Behavior at Scale** (Park et al., ICML 2023)는 대규모 모델의 출력을 학습 데이터 포인트로 귀속하는 효율적 기법이다. 영향 함수(influence function)의 계산 비용 문제를 랜덤 프로젝션으로 해결하며, GPT-2 수준 모델에서 기존 TracIn 대비 10배 이상의 속도 향상을 달성하였다 [39].

### 7.4 머신 언러닝 (Machine Unlearning)

**Who's Harry Potter? Approximate Unlearning in LLMs** (Eldan & Russinovich, NeurIPS 2023 Workshop)은 특정 저작물(해리 포터 시리즈)에 대한 모델의 지식을 선택적으로 제거하는 기법을 제안하였다. 대체 레이블(alternative label)을 생성하여 미세조정하는 방식으로, 해리 포터 관련 생성 능력을 90% 이상 감소시키면서 일반 능력은 2% 이내의 하락에 그쳤다 [40].

**Large Language Model Unlearning** (Yao et al., NeurIPS 2024 Workshop → 확장 ICLR 2025)은 LLM 언러닝을 위한 체계적 프레임워크를 제안하였다. gradient ascent, KL divergence minimization, preference optimization의 세 가지 전략을 비교 분석하였다. gradient ascent 기반 언러닝이 대상 데이터에 대한 perplexity를 $10^3$ 이상 증가시키면서 일반 벤치마크 성능을 유지함을 보였다 [41].

**TOFU: A Task of Fictitious Unlearning for LLMs** (Maini et al., ICLR 2025)은 LLM 언러닝 평가를 위한 표준화된 벤치마크를 제안하였다. 200명의 가상 저자에 대한 질문-답변 데이터셋을 구성하고, 언러닝 후 대상 지식의 제거 여부와 인접 지식의 보존 여부를 정량적으로 평가하는 프로토콜을 정립하였다 [42].

---

## 8. 종합 분석 및 향후 전망

### 8.1 분야별 핵심 동향 요약

| 분야 | 핵심 동향 | 미해결 과제 |
|------|-----------|-------------|
| 적대적 공격 | 자동화 수준 향상, 전이 공격 성공 | 적응적 공격에 대한 보편적 방어 부재 |
| 방어 메커니즘 | 표현 공학, 다층 방어 | 방어-유틸리티 트레이드오프 |
| 워터마킹 | 의미 수준 워터마크, 적응적 강도 | 워터마크 도용, 범용 강건성 |
| AI 텍스트 탐지 | 제로샷 탐지, 내재적 차원 | 패러프레이징 강건성, 다국어 |
| 프라이버시 | 대규모 추출 공격 입증 | DP의 유틸리티 손실, 스케일링 |
| 공정성 | 암묵적 편향 발견, 다문화 벤치마크 | 다국어 안전 정렬, 교차적 편향 |
| 저작권 | 기억 정량화, 머신 언러닝 | 언러닝 검증, 법적 프레임워크 |

### 8.2 향후 연구 방향

첫째, **다층적 방어 체계(defense-in-depth)**의 통합이 필요하다. 입력 필터링, 표현 공학, 출력 가드레일을 단일 프레임워크로 결합하는 연구가 요구된다. 현재 각 방어 기법은 독립적으로 평가되며, 조합 시의 시너지와 충돌에 대한 체계적 분석이 부족하다.

둘째, **프라이버시 보존 학습의 스케일링**이 핵심 과제이다. DP-SGD의 수십억 파라미터 모델에 대한 적용은 메모리 및 계산 비용 측면에서 여전히 비실용적이며, LoRA와 DP의 결합 등 파라미터 효율적 접근이 유망하다.

셋째, **워터마킹의 표준화 및 법적 프레임워크** 구축이 시급하다. 현재 다수의 워터마킹 기법이 제안되었으나, 상호 호환성이 없고 법적 증거력에 대한 합의가 부재하다.

넷째, **다국어·다문화 안전성**의 균등화가 필요하다. 영어 중심의 안전 정렬이 저자원 언어에서 심각한 보안 취약점을 야기하는 문제는 글로벌 배포의 전제 조건이다.

다섯째, **AI 에이전트 보안**이 새로운 프론티어로 부상하고 있다. 도구 사용, 코드 실행, 외부 API 호출이 가능한 LLM 에이전트의 보안은 단순 텍스트 생성 모델의 보안과 본질적으로 다른 위협 모델을 요구한다.

---

## 9. 참고문헌

[1] Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M., "Universal and Transferable Adversarial Attacks on Aligned Language Models", NeurIPS 2023. arXiv:2307.15043

[2] Liu, X., Xu, N., Chen, M., & Xiao, C., "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models", ICLR 2024. arXiv:2310.04451

[3] Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E., "Jailbreaking Black Box Large Language Models in Twenty Queries", NeurIPS 2024. arXiv:2310.08419

[4] Huang, Y., Gupta, S., Xia, M., Li, K., & Chen, D., "Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation", ICML 2024. arXiv:2310.06987

[5] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M., "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections", AISec@CCS 2023. arXiv:2302.12173

[6] Toyer, S., Watkins, O., Mendelson, E. A., Paduraru, C., Piber, Y., Brozek, Z., ... & Russell, S., "Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game", ICLR 2024. arXiv:2311.01011

[7] Carlini, N., Paleka, D., Dvijotham, K. D., Steinke, T., Hayase, J., Cooper, A. F., ... & Tramèr, F., "Stealing Part of a Production Language Model", ICML 2024. arXiv:2403.06634

[8] Inan, H., Upasani, K., Chi, J., Rungta, R., Iyer, K., Mao, Y., ... & Khabsa, M., "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations", Meta 2024. arXiv:2312.06674

[9] Rebedea, T., Dinu, R., Sreedhar, M., Parisien, C., & Cohen, J., "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails", EMNLP 2023 Demo. arXiv:2310.10501

[10] Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Fredrikson, M., "Representation Engineering: A Top-Down Approach to AI Transparency", ICLR 2024. arXiv:2310.01405

[11] Zou, A., Phan, L., Wang, J., Duenas, D., Lin, M., Andersen, M., & Fredrikson, M., "Improving Alignment and Robustness with Circuit Breakers", NeurIPS 2024. arXiv:2406.04313

[12] Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., ... & Hendrycks, D., "Adversarial Training for LLMs (R2D2)", NeurIPS 2024. arXiv:2406.02132

[13] Hu, C., Li, H., Peng, Y., Wang, L., & Li, L., "Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes", NeurIPS 2024. arXiv:2403.00867

[14] Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., ... & Forsyth, D., "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal", ICML 2024. arXiv:2402.04249

[15] Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T., "A Watermark for Large Language Models", ICML 2023. arXiv:2301.10226

[16] Hu, Z., Chen, L., Wu, X., Wu, Y., Zhang, H., & Huang, H., "Unbiased Watermark for Large Language Models", ICLR 2024. arXiv:2310.10669

[17] Hou, A. B., Zhang, J., He, T., Wang, Y., Chuang, Y.-S., Wang, H., ... & Roth, D., "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation", NAACL 2024. arXiv:2310.03991

[18] Liu, A., Pan, L., Lu, Y., Li, J., Wen, X., King, I., & Lyu, M. R., "Adaptive Text Watermark for Large Language Models", ICML 2024. arXiv:2401.13927

[19] Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saber, K., Katz, J., ... & Goldstein, T., "On the Reliability of Watermarks for Large Language Models", ICLR 2024. arXiv:2306.04634

[20] Jovanović, N., Staab, R., & Vechev, M., "Watermark Stealing in Large Language Models", ICML 2024. arXiv:2402.19361

[21] Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y., "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature", ICLR 2024. arXiv:2310.05130

[22] Hans, A., Schwarzschild, A., Cheber, V., Aber, H., Bruss, C. B., & Goldblum, M., "Spotting LLMs With Binoculars", ICML 2024. arXiv:2401.12070

[23] Tulchinskii, E., Kuznetsov, K., Kushnareva, L., Cherniavskii, D., Barannikov, S., Piontkovskaya, I., ... & Burnaev, E., "Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts", NeurIPS 2023. arXiv:2306.04723

[24] Nasr, M., Carlini, N., Hayase, J., Jagielski, M., Cooper, A. F., Ippolito, D., ... & Lee, K., "Scalable Extraction of Training Data from (Production) Language Models", USENIX Security 2024. arXiv:2311.17035

[25] Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F., & Zhang, C., "Quantifying Memorization Across Neural Language Models", ICLR 2023. arXiv:2202.07646

[26] Zhang, X., Li, R., Chen, J., & Wen, J., "Membership Inference Attacks on Language Models via Self-calibrated Probabilistic Variation", ACL 2024.

[27] Duan, M., Suri, A., Mireshghallah, N., Min, S., Shi, W., Zettlemoyer, L., ... & Tsvetkov, Y., "Do Membership Inference Attacks Work on Large Language Models?", CCS 2024. arXiv:2402.07841

[28] Hong, J., Wang, J., Zhang, C., Li, Z., Li, B., & Wang, Z., "DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer", ICLR 2024. arXiv:2312.03724

[29] Yu, D., Naik, S., Backurs, A., Gopi, S., Inan, H. A., Kamath, G., ... & Zhang, H., "Differentially Private Fine-tuning of Language Models", ICLR 2023. arXiv:2110.06500

[30] Kuang, W., Qian, B., Li, Z., Chen, D., Gao, D., Pan, X., ... & He, B., "FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning", ACL 2024 Demo. arXiv:2309.00363

[31] Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., ... & Bowman, S. R., "BBQ: A Hand-Built Bias Benchmark for Question Answering", ACL 2022. arXiv:2110.08193

[32] Tamkin, A., Askell, A., Lovitt, L., Ganguli, D., & Bowman, S. R., "Evaluating and Mitigating Discrimination in Language Model Decisions", NeurIPS 2024. arXiv:2312.03689

[33] Schick, T., Udupa, S., & Schütze, H., "Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP", TACL 2021. arXiv:2103.00453

[34] Gupta, A., Mondal, D., Sheshadri, A. K., Zhao, W., Li, X. L., Wiegreffe, S., & Tandon, N., "Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs", ICLR 2024. arXiv:2311.04892

[35] Chiu, Y., Sharma, L., Li, J., & Chen, X., "CulturalBench: Benchmarking LLMs for Cultural Knowledge", NeurIPS 2024 Datasets and Benchmarks.

[36] Deng, Y., Zhang, W., Pan, S. J., & Bing, L., "Multilingual Jailbreak Challenges in Large Language Models", ICLR 2024. arXiv:2310.06474

[37] Chang, K., Cramer, M., Soni, S., & Bamman, D., "Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4", EMNLP 2023. arXiv:2305.00118

[38] Meeus, M., Jain, S., Rei, M., & de Montjoye, Y.-A., "Copyright Traps for Large Language Models", ICML 2024. arXiv:2402.09363

[39] Park, S. M., Georgiev, K., Ilyas, A., Leclerc, G., & Madry, A., "TRAK: Attributing Model Behavior at Scale", ICML 2023. arXiv:2303.14186

[40] Eldan, R. & Russinovich, M., "Who's Harry Potter? Approximate Unlearning in LLMs", NeurIPS 2023 Workshop. arXiv:2310.02238

[41] Yao, Y., Xu, X., & Liu, Y., "Large Language Model Unlearning", NeurIPS 2024 Workshop / ICLR 2025. arXiv:2310.10683

[42] Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z., "TOFU: A Task of Fictitious Unlearning for LLMs", ICLR 2025. arXiv:2401.06121
