# LLM 환각(Hallucination) 탐지 및 완화 최신 연구 동향 (2024-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, AAAI) 중심
> 총 참고논문: 38편

---

## 1. 개요

대규모 언어 모델(LLM)의 환각(hallucination)은 모델이 사실에 부합하지 않거나 입력 컨텍스트와 모순되는 내용을 마치 사실인 것처럼 생성하는 현상을 지칭한다. 이는 LLM의 실제 배포에서 가장 심각한 신뢰성 문제 중 하나이며, 의료, 법률, 금융 등 사실적 정확성이 요구되는 도메인에서 치명적인 결과를 초래할 수 있다.

환각의 원인은 다층적이다. 사전학습 데이터의 편향과 노이즈, 디코딩 과정에서의 확률적 샘플링, 미세조정 시 새로운 지식의 도입으로 인한 기존 지식과의 충돌, 그리고 모델이 자신의 지식 경계를 인식하지 못하는 과신(overconfidence) 등이 복합적으로 작용한다.

본 문서에서는 (1) 환각 탐지, (2) 학습 시 환각 완화, (3) RAG 기반 환각 감소, (4) 자기 일관성 및 검증, (5) 사실성 평가 벤치마크, (6) 추론 기반 환각 감소의 6개 분야로 나누어 최신 연구를 정리한다.

---

## 2. 환각 탐지 (Hallucination Detection)

### 2.1 기법 등장 배경

환각을 탐지하는 것은 완화의 전제 조건이다. 초기 접근법은 외부 지식 소스와의 교차 검증에 의존하였으나, 이는 높은 비용과 도메인 제약이 존재하였다. 이에 따라 모델의 내부 상태(hidden states), 출력 일관성, 불확실성 정량화 등을 활용한 자동화된 탐지 기법이 등장하였다.

### 2.2 내부 표현 기반 탐지

**HaloScope** (Du et al., NeurIPS 2024 Spotlight)은 레이블이 없는 LLM 생성물을 활용하여 환각을 탐지하는 기법이다. 활성화 공간에서 특이값 분해(SVD)를 통해 환각과 연관된 부분공간을 식별한다. TruthfulQA에서 지도 학습 상한(81.04%)에 근접한 78.64%의 성능을 인간 주석 없이 달성하였다 [1].

**INSIDE** (Chen et al., ICLR 2024)는 EigenScore 메트릭을 제안하여, 조밀 임베딩 공간에서 응답 공분산 행렬의 고유값을 활용해 응답 자기 일관성을 평가한다. 또한 테스트 시 특징 클리핑(feature clipping) 접근법을 도입하여 과신 생성을 줄이고 환각 탐지를 개선한다 [2].

**LLM-Check** (Sriramanan et al., NeurIPS 2024)는 내부 LLM 표현의 고유값 분석과 출력 토큰 불확실성 정량화를 결합한 포괄적 환각 탐지 연구이다. 기존 기준선 대비 최대 450배의 속도 향상과 유의미한 탐지 성능 개선을 달성하였다 [3].

**In-Context Sharpness** (Chen et al., ICML 2024)는 올바른 생성이 환각보다 은닉 상태에서 더 날카로운(sharper) 컨텍스트 활성화를 보인다는 사실을 발견하였다. 엔트로피 기반 메트릭으로 이를 정량화하고 디코딩에 통합하여 TruthfulQA에서 최대 8.6 절대점(absolute points) 향상을 달성하였다 [4].

**Latent Space Chain-of-Embedding** (Wang et al., ICLR 2025)는 단일 생성의 모든 점진적 은닉 상태(Chain-of-Embedding)를 LLM의 "사고 경로"로 활용하는 경량 레이블 프리 자기 평가 기법이다. 7개 LLM에 걸쳐 밀리초 수준의 연산 비용으로 실시간 환각 탐지를 달성한다 [5].

**ReDeEP** (Sun et al., ICLR 2025 Spotlight)는 기계적 해석가능성(mechanistic interpretability)을 활용하여 LLM의 파라메트릭 지식과 외부 컨텍스트 활용을 분리한다. 어텐션 헤드를 통한 External Context Score와 FFN을 통한 Parametric Knowledge Score를 도입하고, Knowledge FFN과 Copying Head를 조절하여 환각을 완화하는 AARF를 제안한다 [6].

### 2.3 출력 기반 탐지

**Semantic Entropy Probes** (Kossen et al., ICML 2024)는 단일 생성의 은닉 상태에서 의미 엔트로피(semantic entropy)를 직접 근사하여, 테스트 시 다중 출력 샘플링의 필요를 제거한다. 의미 엔트로피는 의미적으로 동등한 응답들을 클러스터 $C$로 묶은 후, 클러스터 수준에서 엔트로피를 계산한다:

$$SE(x) = -\sum_{c \in \mathcal{C}} P(c | x) \log P(c | x), \quad P(c | x) = \sum_{s \in c} P(s | x)$$

여기서 $P(s | x)$는 프롬프트 $x$에 대한 개별 응답 $s$의 생성 확률이며, 의미적으로 동등한 응답들이 동일 클러스터 $c$에 속한다. 높은 의미 엔트로피는 모델이 의미적으로 다양한 응답을 생성하므로 환각 가능성이 높음을 나타낸다. 연산 오버헤드를 거의 0으로 줄이면서 강건한 환각 탐지 성능을 유지한다 [7].

**Kernel Language Entropy** (Hofmann et al., NeurIPS 2024)는 양정치 반정부호(positive semidefinite) 커널로 의미적 유사성을 인코딩하고, von Neumann 엔트로피로 불확실성을 정량화하는 KLE를 제안한다. 의미 엔트로피를 이론적으로 일반화하고 다수의 NLG 데이터셋에서 불확실성 정량화를 개선한다 [8].

**SelfCheckGPT** (Manakul et al., EMNLP 2023)는 제로 리소스, 블랙박스 환각 탐지 기법으로, 알려진 사실에 대한 샘플링 응답은 일관되지만 환각된 사실은 샘플 간 발산한다는 통찰에 기반한다. BERTScore 변형의 경우, 원본 응답의 문장 $s_i$에 대한 환각 점수는 다음과 같이 산출된다:

$$\text{SelfCheck}(s_i) = 1 - \frac{1}{N} \sum_{j=1}^{N} \max_{s' \in R_j} \text{BERTScore}(s_i, s')$$

여기서 $R_j$는 $j$번째 샘플링된 응답의 문장 집합이며, $N$은 샘플 수이다. 점수가 높을수록 해당 문장이 다른 샘플들과 일관되지 않아 환각일 가능성이 높다. 문장 수준 환각 탐지에서 강한 AUC-PR 점수를 보였다 [9].

**Self-Contradictory Hallucinations** (Mundler et al., ICLR 2024)는 블랙박스 LM에 적용 가능한 프롬프팅 전략을 통해 자기 모순을 유발, 탐지, 완화하는 알고리즘을 개발하였다. ChatGPT 자기 모순의 63.1%에서 모순되는 두 문장 모두 비사실적임을 밝혔다 [10].

---

## 3. 학습 시 환각 완화 (Training-time Mitigation)

### 3.1 기법 등장 배경

표준 SFT(Supervised Fine-Tuning)와 RLHF/DPO 정렬 과정이 의도치 않게 환각을 조장할 수 있다는 연구 결과가 보고되면서, 학습 단계에서의 사실성 인식(factuality-aware) 접근법이 등장하였다. 특히 미세조정 데이터에 모델이 사전학습에서 습득하지 못한 새로운 지식이 포함될 경우, 모델의 환각 경향이 선형적으로 증가한다는 발견이 이 분야의 핵심 동기이다.

### 3.2 사실성 인식 정렬 (Factuality-Aware Alignment)

**FLAME** (Lin et al., NeurIPS 2024)은 표준 SFT+DPO 정렬이 모델에 친숙하지 않은 데이터로 학습하고 긴 응답을 선호하는 보상 함수를 사용함으로써 환각을 부추긴다는 사실을 규명하였다. 사실성 인식 DPO의 손실 함수는 다음과 같이 정의된다:

$$\mathcal{L}_{\text{Fact-DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

여기서 $(y_w, y_l)$은 FActScore에 의해 사실성이 높은/낮은 응답 쌍으로 구성되며, $\beta$는 KL 페널티 계수이다. FActScore 기반 필터링을 통한 사실성 인식 SFT와 사실성 인식 DPO를 제안하여, 명령 수행 능력을 유지하면서 사실성을 개선한다 [11].

**Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?** (Gekhman et al., EMNLP 2024)은 새로운 지식을 도입하는 미세조정 예제가 기존 지식과 일치하는 예제보다 유의미하게 느리게 학습되며, 이러한 예제가 학습됨에 따라 모델의 환각 경향이 선형적으로 증가한다는 사실을 밝혔다. LLM이 사실적 지식을 주로 사전학습을 통해 습득한다는 견해를 뒷받침한다 [12].

**FactAlign** (Huang et al., EMNLP 2024 Findings)은 Kahneman-Tversky Optimization(KTO)을 확장한 세분화된 문장 수준 정렬 알고리즘 fKTO를 도입한다. KTO의 가치 함수(value function)는 Kahneman-Tversky의 전망 이론에 기반하여 다음과 같이 정의된다:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x,y)}\left[\lambda_y \cdot \sigma\left(\beta \cdot \left(r_\theta(x, y) - z_{\text{ref}}\right)\right)\right]$$

여기서 $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$는 로그 비율, $z_{\text{ref}}$는 KL 기준점, $\lambda_y$는 바람직한/비바람직한 응답에 대한 비대칭 가중치이다. fKTO는 이를 문장 수준으로 확장하여 각 원자적 사실의 정확성에 따라 가중치를 차별화한다. 자동 사실성 평가를 활용하여 정렬을 유도하며, 사실적 정확도와 유용성, 사실적 F1 점수를 모두 개선한다 [13].

### 3.3 거부 학습 및 지식 인식

**R-Tuning** (Zhang et al., NAACL 2024 Outstanding Paper Award)은 사전학습 파라미터와 명령 조정 데이터 간의 지식 격차를 식별하고, 거부 인식 데이터를 구성하여 LLM이 파라메트릭 지식을 초과하는 질문에 "모르겠다"라고 답하도록 학습시킨다. 거부 능력이 미지의 태스크로 일반화되는 메타 스킬로 작동함을 보였다 [14].

**Do I Know This Entity?** (Ferrando et al., ICLR 2025 Oral)는 희소 오토인코더(sparse autoencoders)를 활용하여 핵심 환각 메커니즘이 개체 인식(entity recognition)임을 발견하였다. 표현 공간에서 자기 지식을 인코딩하는 선형 방향이 존재하며, 이것이 채팅 모델의 거부 행동에 인과적으로 영향을 미침을 밝혔다 [15].

**Unfamiliar Finetuning Examples** (Kang et al., NAACL 2025)는 LLM의 환각 예측이 미숙한 미세조정 예제와 연관된 응답을 반영한다는 사실을 보이고, 미숙한 예제의 감독을 수정함으로써 미숙한 쿼리에 대한 모델 응답에 영향을 줄 수 있음을 입증하였다 [16].

**HaDeMiF** (Chen et al., ICLR 2025)는 출력 및 의미 공간에서 환각을 포착하는 고급 프레임워크로, 해석 가능한 환각 탐지를 위한 Deep Dynamic Decision Tree(D3T)와 의미 수준 환각을 포착하는 MLP를 결합한다 [17].

---

## 4. RAG 기반 환각 감소

### 4.1 기법 등장 배경

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 외부 지식을 활용하여 LLM의 사실적 근거를 강화하는 대표적 접근법이다. 그러나 검색된 문서의 품질이 낮거나 관련 없는 경우, RAG 자체가 환각을 유발할 수 있다는 한계가 지적되면서, 이를 극복하기 위한 자기 반성(self-reflection) 및 품질 평가 메커니즘이 발전하였다.

**Self-RAG** (Asai et al., ICLR 2024 Oral, top 1%)는 단일 LM을 학습시켜 필요에 따라 적응적으로 문서를 검색하고, 특수 반성 토큰(reflection tokens)을 생성하여 자체 생성의 품질과 사실성을 자기 평가한다. Self-RAG(7B/13B)는 QA, 추론, 사실 검증 태스크에서 ChatGPT와 검색 증강 Llama2-chat을 유의미하게 상회하였다 [18].

**RAG-HAT** (EMNLP 2024 Industry Track)는 환각 탐지 모델을 학습시키고, 환각 설명을 활용하여 GPT-4 Turbo로 오류를 교정한다. DPO 학습을 위한 선호 데이터셋을 생성하여 RAG 환경에서 환각률을 낮추고 답변 품질을 개선한다 [19].

**CRAG (Corrective Retrieval Augmented Generation)** (Yan et al., 2024)는 경량 검색 평가기를 설계하여 문서 품질을 평가하고, 신뢰도에 따라 서로 다른 지식 검색 행동을 트리거한다. 웹 검색 증강과 분해-재조합 알고리즘을 통해 핵심 정보에 선택적으로 집중하고 무관한 내용을 필터링한다 [20].

**Reducing Hallucination in Structured Outputs via RAG** (NAACL 2024 Industry Track)는 RAG를 활용하여 구조화된 출력의 품질을 향상시키고, 도메인 외(out-of-domain) 환경에서의 환각 감소와 일반화 개선을 달성하였다 [21].

**Knowledge Graphs for LLM Hallucination Reduction** (Agrawal et al., NAACL 2024)는 LLM을 위한 지식 그래프 기반 증강 기법을 포괄적으로 검토하고, 환각 완화를 위한 세 가지 범주로 방법론을 분류하여 성능 비교를 제공한다 [22].

---

## 5. 자기 일관성 및 검증 (Self-Consistency and Verification)

### 5.1 기법 등장 배경

환각된 내용은 반복 생성 시 일관성을 보이지 않는다는 관찰에 기반하여, 모델 스스로가 자신의 출력을 검증하고 교정하는 접근법이 등장하였다. 이는 외부 지식 소스 없이도 작동할 수 있다는 장점이 있다.

**Chain-of-Verification (CoVe)** (Dhuliawala et al., ACL 2024 Findings)은 4단계 파이프라인을 제안한다: (1) 초안 응답 생성, (2) 검증 질문 계획, (3) 독립적으로 검증 질문 답변, (4) 최종 검증 응답 생성. 검증을 분리(decoupled verification)함으로써 QA 및 장문 생성 벤치마크에서 사실적 환각을 50-70% 감소시켰다 [23].

**SELF-FAMILIARITY** (Luo et al., EMNLP 2024 Findings)는 모델이 입력 개념에 대한 친숙도를 자기 평가하고, 미숙한 개념에 대해서는 응답 생성을 보류하는 사전 탐지 기법이다. 4개 LLM에서 일관되게 우수한 환각 방지 성능을 보였다 [24].

**Law of Knowledge Overshadowing** (Zhang et al., ACL 2025 Findings)은 환각률이 지식 인기도, 지식 길이, 모델 크기의 로그 스케일에 선형적으로 증가한다는 로그-선형 법칙을 도입하였다. CoDA 디코딩 전략을 제안하여 Overshadow 벤치마크에서 사실성을 27.9%, NQ-Swap에서 18.3% 개선하였다 [25].

**CDT (Contrasting Decoding with Truthful comparators)** (Zhang et al., AAAI 2025)는 다중 태스크 미세조정을 통해 환각 및 사실적 비교자를 구성하고, Mixture of Experts 전략으로 서로 다른 환각 패턴을 포착한다. 로짓 차이를 대조하여 다음 토큰 예측을 사실성에 강건한 분포로 제약한다 [26].

---

## 6. 사실성 평가 벤치마크 (Factuality Evaluation Benchmarks)

### 6.1 기법 등장 배경

환각 완화 연구의 진전을 위해서는 신뢰할 수 있는 평가 기준이 필수적이다. 초기 벤치마크는 폐쇄형 QA에 국한되었으나, 최근에는 개방형 장문 생성, 동적 테스트셋, 다차원 평가를 지원하는 벤치마크가 개발되고 있다.

**SAFE (Search-Augmented Factuality Evaluator)** (Wei et al., Google DeepMind, NeurIPS 2024)는 LLM 에이전트를 사용하여 응답을 원자적 사실(atomic facts)로 분해하고 Google 검색을 통해 각각을 검증한다. LongFact 벤치마크(38개 주제, 수천 질문)를 생성하였다. 인간 주석자와 72% 일치하면서 비용은 20배 저렴하다 [27].

**FActScore** (Min et al., EMNLP 2023)은 생성 텍스트를 원자적 사실로 분해하고, 신뢰할 수 있는 지식 소스에 의해 지지되는 비율을 계산한다. FActScore는 다음과 같이 정의된다:

$$\text{FActScore}(y) = \frac{1}{|\mathcal{A}(y)|} \sum_{a \in \mathcal{A}(y)} \mathbb{1}[\text{supported}(a, \mathcal{K})]$$

여기서 $y$는 생성 텍스트, $\mathcal{A}(y)$는 $y$에서 추출한 원자적 사실의 집합, $\mathcal{K}$는 지식 소스(예: Wikipedia)이며, $\text{supported}(a, \mathcal{K})$는 원자적 사실 $a$가 $\mathcal{K}$에 의해 지지되는지 여부를 판정한다. ChatGPT조차 인물 전기에서 58%의 사실적 정밀도만 달성함을 보였으며, 후속 사실성 연구의 기초 평가 프레임워크를 확립하였다 [28].

**FactBench** (Bayat et al., ACL 2025)는 웹 검색 증거 기반으로 내용 단위를 Supported, Unsupported, Undecidable로 분류하는 VERIFY 파이프라인을 도입한다. 150개 주제 3개 난이도 계층의 1K 프롬프트를 제공하며, 사실성이 모델 규모에 비례하여 개선되지 않음을 발견하였다 [29].

**HalluLens** (Facebook Research, ACL 2025)는 외재적(extrinsic) 및 내재적(intrinsic) 환각을 구분하는 분류체계를 제안하고, LongWiki, PreciseQA, Nonsense의 세 가지 평가 태스크를 도입한다. 데이터 누출 방지를 위한 동적 테스트셋 생성을 지원한다 [30].

**HALoGEN** (Ravichander et al., ACL 2025)는 9개 도메인 10,923개 프롬프트와 자동 검증기를 포함하는 포괄적 벤치마크이다. 14개 LLM의 약 150,000개 생성물을 평가한 결과, 일부 도메인에서 원자적 사실의 최대 86%가 환각임을 발견하였다. 환각을 Type A(잘못된 회상), Type B(잘못된 학습 데이터), Type C(조작)로 분류한다 [31].

**HalluEditBench** (Huang et al., ICLR 2025)는 9개 도메인 26개 주제에 걸친 6,000개 이상의 환각으로 대규모 벤치마크를 구축하여 지식 편집 방법을 5개 차원(Efficacy, Generalization, Portability, Locality, Robustness)으로 평가한다. 지식 편집의 효과가 기존 벤치마크가 제시하는 것보다 훨씬 부족함을 밝혔다 [32].

**Factuality of LLMs: A Survey** (Wang et al., EMNLP 2024)는 LLM 사실성에 관한 기존 연구를 비판적으로 분석하고, 주요 과제와 그 원인을 식별하며, 개방형 텍스트 생성에 대한 자동화된 사실성 평가의 장애물을 분석한다 [33].

---

## 7. 추론 기반 환각 감소 (Reasoning for Hallucination Reduction)

### 7.1 기법 등장 배경

Chain-of-Thought(CoT) 프롬프팅이 추론 성능을 향상시키는 것으로 알려져 있으나, 이것이 환각에 미치는 영향은 양면적이다. CoT가 환각 빈도를 줄이는 동시에 환각 탐지를 위한 핵심 신호를 가린다는 트레이드오프가 발견되면서, 추론과 환각의 관계에 대한 심층적 연구가 진행되고 있다.

**CoT Obscures Hallucination Cues** (Yao et al., EMNLP 2025 Findings)는 CoT 프롬프팅이 환각 빈도를 줄이는 데 도움이 되지만, 동시에 환각 탐지에 사용되는 핵심 신호를 가려 신뢰할 수 있는 탐지기 구축을 어렵게 한다는 중요한 트레이드오프를 밝혔다 [34].

**Bottom-Up Holistic Reasoning** (Wu et al., AAAI 2025)는 인간의 직관에서 영감을 받은 상향식 추론 프레임워크로, 장면 그래프 표현을 활용하여 이미지 이해를 강화한다. 지각 수준 정보와 인지 수준 상식 지식을 체계적으로 검증하고 통합하여 더 신뢰할 수 있는 멀티모달 출력을 생성한다 [35].

**Attributive Reasoning for Hallucination Diagnosis** (Liu et al., AAAI 2025)는 LLM 내부 신호에 기반한 환각 원인 추적을 위한 귀인 프레임워크를 도입하고, 8개 환각 범주를 포함하는 RelQA-Cate 벤치마크를 개발한다. Differential Penalty Decoding(DPD)을 제안하여 답변 신뢰성을 최대 28.25% 개선한다 [36].

**HaMI (Hallucination Mitigation via Adaptive Token Selection)** (Li et al., NeurIPS 2025)는 환각 탐지기가 사전 결정된 토큰 위치에 민감한 문제를 해결하기 위해, 환각을 가장 잘 나타내는 핵심 토큰을 적응적으로 선택하고 학습하는 기법을 제안한다. 다양한 길이와 희소한 환각 개체 분포를 가진 자유형 생성에서 강건한 탐지를 보인다 [37].

**Hallucination-Induced Optimization** (Yu et al., NeurIPS 2024)는 Contrary Bradley-Terry Model을 활용하여 환각 토큰과 목표 토큰 간의 대비를 증폭시키는 새로운 최적화 전략을 도입한다. 효율적 대비 디코딩을 촉진하여 대형 시각-언어 모델(LVLM)에서의 환각을 완화한다 [38].

---

## 8. 연구 동향 종합 분석

### 8.1 탐지에서 예방으로의 패러다임 전환

초기 연구가 생성 후 환각을 탐지하는 데 초점을 맞추었다면, 최근에는 학습 단계에서 사실성을 내재화하고, 디코딩 단계에서 사실성에 강건한 분포를 생성하며, 모델이 자신의 지식 한계를 인식하여 거부하도록 학습시키는 예방적 접근법이 주류가 되고 있다.

### 8.2 해석가능성과 환각의 접점

기계적 해석가능성(mechanistic interpretability) 연구의 발전으로, 어텐션 헤드, FFN, 특정 뉴런 수준에서 환각의 원인을 규명하는 연구가 활발해지고 있다. 이는 환각의 근본적 원인 이해와 인과적 완화를 가능하게 한다.

### 8.3 벤치마크의 정교화

단순 QA를 넘어 장문 생성, 구조화된 출력, 멀티모달 환각, 지식 편집 효과 등을 평가하는 다차원적 벤치마크가 등장하고 있다. 데이터 누출 방지를 위한 동적 테스트셋 생성도 주요 추세이다.

---

## 9. 참고문헌

[1] Du, X. et al. "HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection." NeurIPS 2024 (Spotlight). https://arxiv.org/abs/2409.17504

[2] Chen, C. et al. "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection." ICLR 2024. https://arxiv.org/abs/2402.03744

[3] Sriramanan, G. et al. "LLM-Check: Investigating Detection of Hallucinations in Large Language Models." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/3c1e1fdf305195cd620c118aaa9717ad-Abstract-Conference.html

[4] Chen, S. et al. "In-Context Sharpness as Alerts: An Inner Representation Perspective for Hallucination Mitigation." ICML 2024. https://arxiv.org/abs/2403.01548

[5] Wang, Y. et al. "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation." ICLR 2025. https://arxiv.org/abs/2410.13640

[6] Sun, Z. et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." ICLR 2025 (Spotlight). https://arxiv.org/abs/2410.11414

[7] Kossen, J. et al. "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs." ICML 2024. https://arxiv.org/abs/2406.15927

[8] Hofmann, D. et al. "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities." NeurIPS 2024. https://arxiv.org/abs/2405.20003

[9] Manakul, P. et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." EMNLP 2023. https://arxiv.org/abs/2303.08896

[10] Mundler, N. et al. "Self-Contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation." ICLR 2024. https://arxiv.org/abs/2305.15852

[11] Lin, S.-C. et al. "FLAME: Factuality-Aware Alignment for Large Language Models." NeurIPS 2024. https://arxiv.org/abs/2405.01525

[12] Gekhman, Z. et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" EMNLP 2024. https://arxiv.org/abs/2405.05904

[13] Huang, C.-W. et al. "FactAlign: Long-form Factuality Alignment of Large Language Models." EMNLP 2024 (Findings). https://aclanthology.org/2024.findings-emnlp.955/

[14] Zhang, H. et al. "R-Tuning: Instructing Large Language Models to Say 'I Don't Know'." NAACL 2024 (Outstanding Paper Award). https://arxiv.org/abs/2311.09677

[15] Ferrando, J. et al. "Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models." ICLR 2025 (Oral). https://arxiv.org/abs/2411.14257

[16] Kang, K. et al. "Unfamiliar Finetuning Examples Control How Language Models Hallucinate." NAACL 2025. https://arxiv.org/abs/2403.05612

[17] Chen, Z. et al. "HaDeMiF: Hallucination Detection and Mitigation in Large Language Models." ICLR 2025. https://openreview.net/forum?id=VwOYxPScxB

[18] Asai, A. et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024 (Oral). https://arxiv.org/abs/2310.11511

[19] "RAG-HAT: A Hallucination-Aware Tuning Pipeline for LLM in Retrieval-Augmented Generation." EMNLP 2024 (Industry Track). https://aclanthology.org/2024.emnlp-industry.113/

[20] Yan, S.-Q. et al. "Corrective Retrieval Augmented Generation (CRAG)." arXiv 2024. https://arxiv.org/abs/2401.15884

[21] "Reducing Hallucination in Structured Outputs via Retrieval-Augmented Generation." NAACL 2024 (Industry Track). https://arxiv.org/abs/2404.08189

[22] Agrawal, G. et al. "Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey." NAACL 2024. https://arxiv.org/abs/2311.07914

[23] Dhuliawala, S. et al. "Chain-of-Verification Reduces Hallucination in Large Language Models." ACL 2024 (Findings). https://arxiv.org/abs/2309.11495

[24] Luo, J. et al. "Zero-Resource Hallucination Prevention for Large Language Models." EMNLP 2024 (Findings). https://aclanthology.org/2024.findings-emnlp.204/

[25] Zhang, Y. et al. "The Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination." ACL 2025 (Findings). https://arxiv.org/abs/2502.16143

[26] Zhang, L. et al. "Improving Factuality in Large Language Models via Decoding-Time Hallucinatory and Truthful Comparators." AAAI 2025. https://arxiv.org/abs/2408.12325

[27] Wei, J. et al. "Long-form Factuality in Large Language Models (SAFE + LongFact)." NeurIPS 2024. https://arxiv.org/abs/2403.18802

[28] Min, S. et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023. https://arxiv.org/abs/2305.14251

[29] Bayat, F. F. et al. "FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation." ACL 2025. https://arxiv.org/abs/2410.22257

[30] Chen, T. et al. "HalluLens: LLM Hallucination Benchmark." ACL 2025. https://arxiv.org/abs/2504.17550

[31] Ravichander, A. et al. "HALoGEN: Fantastic LLM Hallucinations and Where to Find Them." ACL 2025. https://arxiv.org/abs/2501.08292

[32] Huang, B. et al. "Can Knowledge Editing Really Correct Hallucinations? (HalluEditBench)." ICLR 2025. https://arxiv.org/abs/2410.16251

[33] Wang, Y. et al. "Factuality of Large Language Models: A Survey." EMNLP 2024. https://arxiv.org/abs/2402.02420

[34] Yao, S. et al. "Chain-of-Thought Prompting Obscures Hallucination Cues in Large Language Models." EMNLP 2025 (Findings). https://arxiv.org/abs/2506.17088

[35] Wu, S. et al. "Combating Multimodal LLM Hallucination via Bottom-Up Holistic Reasoning." AAAI 2025. https://arxiv.org/abs/2412.11124

[36] Liu, X. et al. "Attributive Reasoning for Hallucination Diagnosis of Large Language Models." AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/34536

[37] Li, R. et al. "Robust Hallucination Detection in LLMs via Adaptive Token Selection (HaMI)." NeurIPS 2025. https://arxiv.org/abs/2504.07863

[38] Yu, Q. et al. "Alleviating Hallucinations in Large Vision-Language Models through Hallucination-Induced Optimization." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/dde040998d82553cf7f689e8ae173d5a-Paper-Conference.pdf
