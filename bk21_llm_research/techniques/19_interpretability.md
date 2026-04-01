# LLM 해석가능성, 지식편집 및 지속학습 최신 연구 동향 (2023-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI) 중심
> 총 참고논문: 42편

---

## 1. 개요

대규모 언어 모델(LLM)의 성능이 비약적으로 향상됨에 따라, 모델 내부에서 어떤 계산이 이루어지는지 이해하고(해석가능성), 학습된 지식을 효율적으로 수정하며(지식편집), 새로운 지식을 기존 능력의 훼손 없이 습득하는(지속학습) 세 가지 연구 분야가 핵심적으로 부상하였다. 이 세 분야는 서로 밀접하게 연결되어 있다. 해석가능성 연구가 모델의 내부 지식 표현 방식을 밝힘으로써 지식편집의 정밀도를 높이고, 지속학습은 지식편집을 대규모로 확장하는 형태로 발전하며, 모델 병합 기법은 지속학습의 망각 문제를 우회하는 대안적 접근법으로 기능한다.

본 문서에서는 (1) 기계적 해석가능성(회로 발견, 희소 오토인코더), (2) 프로빙 및 내부 표현 이해, (3) 지식 편집, (4) 지속적/평생 학습, (5) 모델 병합, (6) 창발적 능력 및 위상 전이의 6개 분야로 나누어 2023-2025년 탑 컨퍼런스 중심의 최신 연구를 체계적으로 정리한다.

---

## 2. 기계적 해석가능성 (Mechanistic Interpretability)

### 2.1 기법 등장 배경

신경망의 해석가능성 연구는 오래전부터 존재하였으나, 기존 접근법은 주로 사후적(post-hoc) 설명—예를 들어 attention weight 시각화, LIME, SHAP 등—에 의존하였다. 이러한 방법들은 모델이 "왜" 특정 출력을 생성하는지에 대한 인과적 설명을 제공하지 못한다는 근본적 한계가 존재하였다. Elhage et al. (2021)의 연구 이후, 모델 내부의 개별 구성요소(뉴런, 어텐션 헤드, MLP 레이어)가 수행하는 구체적인 계산을 역공학(reverse-engineering)하는 "기계적 해석가능성(mechanistic interpretability)" 패러다임이 등장하였다.

그러나 개별 뉴런 수준의 분석은 다의성(polysemanticity) 문제에 직면하였다. 하나의 뉴런이 여러 개념을 동시에 인코딩하고, 하나의 개념이 여러 뉴런에 분산 인코딩되는 현상으로 인해, 뉴런 단위의 해석은 신뢰성이 낮았다. 이를 해결하기 위해 희소 오토인코더(Sparse Autoencoder, SAE)를 활용하여 다의적 뉴런을 단의적(monosemantic) 특징으로 분해하는 접근법이 제안되었다. 또한 회로 발견(circuit discovery)은 특정 행동을 담당하는 최소한의 모델 부분 구조를 찾아내는 방향으로 발전하였다.

### 2.2 희소 오토인코더 (Sparse Autoencoders)

**Scaling Monosemanticity** (Templeton et al., 2024)는 Anthropic이 Claude 3 Sonnet에 대해 수백만 개의 특징을 추출한 대규모 SAE 연구이다. SAE는 모델의 잔차 스트림(residual stream) 활성화 $\mathbf{x} \in \mathbb{R}^d$를 희소 코드 $\mathbf{z} \in \mathbb{R}^m$ ($m \gg d$)로 인코딩한다:

$$\mathbf{z} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \quad \hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$$

손실 함수는 재구성 오차와 L1 희소성 페널티의 합으로 구성된다:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{z}\|_1$$

3400만 개의 특징을 학습하여 도시, 프로그래밍 개념, 안전 관련 개념 등 해석 가능한 추상적 특징들을 발견하였다. 특징의 활성화를 인위적으로 조절(steering)함으로써 모델 행동을 인과적으로 변경할 수 있음을 입증하였다 [1].

**Scaling and Evaluating Sparse Autoencoders** (Gao et al., ICLR 2025)는 TopK SAE 아키텍처를 제안하여 기존 L1 정규화 기반 SAE의 한계를 극복하였다. TopK 활성화 함수는 상위 k개의 잠재 차원만 활성화하여 정확한 희소성 수준을 보장한다:

$$\mathbf{z} = \text{TopK}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}})$$

GPT-4 수준 모델에서 1600만 잠재 차원까지 확장하여 파레토 최적 성능을 달성하였다. 평가 지표로 다운스트림 손실(downstream loss), L0 희소성, 탐지 확률(probe loss)의 세 가지를 제안하였다 [2].

**Sparse Autoencoders Find Highly Interpretable Linguistic Features in Language Models** (Cunningham et al., ICLR 2024)는 SAE가 개별 뉴런보다 유의미하게 더 해석 가능한 특징을 추출함을 최초로 체계적으로 검증한 연구이다. GPT-2에서 추출한 특징의 해석 가능성을 인간 평가자를 통해 측정한 결과, SAE 특징의 93%가 해석 가능한 반면, 개별 뉴런은 32%에 그쳤다 [3].

**Transcoders** (Dunefsky et al., NeurIPS 2024)는 SAE를 MLP 서브레이어에 특화하여 적용하는 트랜스코더(transcoder) 아키텍처를 제안하였다. 기존 SAE가 잔차 스트림의 활성화를 분해하는 반면, 트랜스코더는 MLP 입력 $\mathbf{x}_{\text{in}}$을 받아 MLP 출력 $\mathbf{y}_{\text{out}}$을 직접 근사한다. 이를 통해 MLP 계층의 계산을 해석 가능한 구성요소로 분해하고, 회로 분석 시 정보 흐름을 더 명확하게 추적할 수 있다 [4].

### 2.3 회로 발견 (Circuit Discovery)

**Attribution Patching** (Nanda et al., 2023; Syed et al., NeurIPS 2024)은 대규모 모델에서 특정 행동을 담당하는 회로를 효율적으로 발견하기 위한 기법이다. 전체적 활성화 패칭(activation patching)은 각 구성요소에 대해 전방향 패스를 반복해야 하므로 $O(n)$의 비용이 소요되는 반면, 귀속 패칭(attribution patching)은 단일 역전파를 통해 모든 구성요소의 중요도를 근사한다:

$$\text{AttrPatch}(c) = (\mathbf{a}_c^{\text{clean}} - \mathbf{a}_c^{\text{corrupt}}) \cdot \nabla_{\mathbf{a}_c} \mathcal{L}\big|_{\text{clean}}$$

여기서 $\mathbf{a}_c^{\text{clean}}$과 $\mathbf{a}_c^{\text{corrupt}}$는 각각 정상 입력과 손상된 입력에서의 구성요소 $c$의 활성화이다. Syed et al.은 이를 SAE 특징 수준으로 확장하여 수백만 개의 특징에 대해 회로 발견을 수행하였다 [5].

**Automatic Circuit Discovery (ACDC)** (Conmy et al., ICLR 2024)는 계산 그래프에서 특정 태스크에 불필요한 간선을 반복적으로 제거하는 자동 회로 발견 알고리즘이다. 각 간선 $(u, v)$에 대해 해당 간선의 활성화를 손상된 값으로 대체한 후 출력 변화가 임계값 $\tau$ 이하이면 해당 간선을 제거한다. Indirect Object Identification, Greater-Than, Docstring 등의 태스크에서 수동 발견 회로와 높은 일치율을 보였다 [6].

**Sparse Feature Circuits** (Marks et al., NeurIPS 2024)는 SAE 특징 수준에서 회로를 발견하는 프레임워크이다. 뉴런이나 어텐션 헤드 수준이 아닌 SAE 특징 수준에서 회로를 구성함으로써, 각 노드가 해석 가능한 의미를 갖는 "특징 회로(feature circuit)"를 구성한다. GPT-2와 Pythia 모델에서 주어-동사 일치, 편향 행동 등의 태스크에 대해 특징 회로를 발견하고, 이를 통한 타겟 편집이 가능함을 보였다 [7].

### 2.4 특징 조향 (Feature Steering)

**Representation Engineering (RepE)** (Zou et al., ICLR 2024)는 모델의 내부 표현에서 개념 벡터를 추출하고, 이를 추론 시 더하거나 빼서 모델의 행동을 조절하는 기법이다. 정직성(honesty), 편향, 감정 등의 고수준 개념에 대한 제어 벡터를 학습하여, 미세조정 없이도 모델 행동을 조절할 수 있다. 안전성 벤치마크에서 RLHF 기반 정렬과 비견되는 성능을 달성하였다 [8].

**Steering Vectors** (Turner et al., ICLR 2024 Workshop)에서는 대조적 입력 쌍으로부터 활성화 차이를 계산하여 조향 벡터를 추출한다. 잔차 스트림의 특정 레이어 $l$에서 조향 벡터 $\mathbf{v}$를 추가하여 $\mathbf{h}_l' = \mathbf{h}_l + \alpha \mathbf{v}$로 수정한다. 계수 $\alpha$를 조절함으로써 조향 강도를 제어할 수 있다 [9].

---

## 3. 프로빙 및 내부 표현 이해

### 3.1 기법 등장 배경

프로빙(probing)은 사전학습된 모델의 내부 표현이 어떤 언어적, 사실적 정보를 인코딩하는지를 분석하는 기법으로, Alain & Bengio (2017)의 선형 분류기 프로빙에서 시작되었다. 초기 프로빙 연구는 주로 구문 정보(품사, 의존 관계) 인코딩 여부를 확인하는 데 집중하였으나, LLM의 규모가 커지면서 사실적 지식, 진실성(truthfulness), 세계 모델(world model) 등 고차원적 정보의 인코딩 여부로 연구가 확장되었다. 그러나 프로빙 분류기 자체가 정보를 학습하는 것인지, 아니면 표현에 이미 존재하는 정보를 읽어내는 것인지에 대한 논란(selectivity 문제)이 지속적으로 제기되었다.

### 3.2 진실성 표현 (Truthfulness Representations)

**Geometry of Truth** (Marks & Tegmark, ICLR 2024)는 LLM의 내부 표현에서 진실 값(truth value)이 선형적으로 인코딩됨을 발견하였다. 사실/반사실 문장 쌍에 대한 활성화를 수집하고 PCA 분석을 수행한 결과, 첫 번째 주성분이 진실성 방향과 높은 상관을 보였다. 이는 LLM이 단순한 통계적 패턴 매칭을 넘어 사실에 대한 내부 표현을 형성함을 시사한다 [10].

**Inference-Time Intervention (ITI)** (Li et al., NeurIPS 2023)은 대조적 활성화 분석을 통해 진실한 응답과 거짓 응답의 활성화 차이를 "진실성 방향"으로 추출하고, 추론 시 특정 어텐션 헤드에 개입하여 진실성을 향상시키는 기법이다. TruthfulQA 벤치마크에서 기본 모델 대비 최대 11.3%의 진실성 향상을 미세조정 없이 달성하였다 [11].

### 3.3 세계 모델 및 공간 표현

**Othello-GPT** (Li et al., ICLR 2023)는 오셀로 게임의 수(move) 시퀀스만으로 학습된 GPT 모델이 내부적으로 보드 상태(world state)를 표현함을 발견한 선구적 연구이다. 비선형 프로빙으로 보드 상태를 97.2% 정확도로 예측할 수 있었으며, 개입 실험을 통해 이 표현이 단순 통계가 아닌 인과적으로 사용됨을 입증하였다 [12].

**Language Models Represent Space and Time** (Gurnee & Tegmark, ICLR 2024)는 Llama-2 모델의 내부 표현에서 공간적, 시간적 정보가 선형적으로 인코딩됨을 밝혔다. 도시의 위도/경도, 역사적 사건의 연도 등이 중간 레이어에서 선형 프로빙으로 높은 정확도($R^2 > 0.9$)로 예측 가능하였다. 이는 LLM이 텍스트 통계를 넘어 현실 세계의 구조적 표현을 학습함을 시사한다 [13].

### 3.4 특징 해석 자동화

**Automated Interpretability** (Bills et al., 2023; Bricken et al., 2023)에서 시작된 자동 해석 파이프라인은 GPT-4 등의 대형 모델을 활용하여 뉴런이나 SAE 특징의 활성화 패턴을 자동으로 설명하는 접근법이다. 그러나 초기 자동 설명의 정확도는 60-70% 수준에 그쳤다.

**FIND (Feature Interpretation with Natural language Descriptions)** (Foote et al., ICLR 2024 Spotlight)는 뉴런의 최대 활성화 예제를 수집하고, 이를 기반으로 대형 모델이 자연어 설명을 생성한 후, 해당 설명이 새로운 예제에서의 활성화를 예측할 수 있는지 검증하는 자동화 파이프라인을 제안하였다. 시뮬레이션 정확도 지표를 통해 설명의 충실도를 정량화한다 [14].

---

## 4. 지식 편집 (Knowledge Editing)

### 4.1 기법 등장 배경

LLM은 사전학습 시점의 데이터에 기반하여 지식을 인코딩하므로, 시간이 지남에 따라 오래된(outdated) 정보를 포함하게 된다. 예를 들어 "영국의 총리는 X이다"라는 사실이 변경되었을 때, 전체 모델을 재학습하는 것은 막대한 계산 비용을 수반한다. 이에 따라 모델의 파라미터를 직접 수정하여 특정 사실만 변경하는 "지식 편집(knowledge editing)" 기법이 등장하였다.

초기 접근법인 KnowledgeEditor (De Cao et al., 2021)와 MEND (Mitchell et al., ICML 2022)는 하이퍼네트워크를 통해 그래디언트 업데이트를 예측하는 방식이었으나, 편집의 국소성(locality)—즉, 관련 없는 지식에 대한 부작용 최소화—이 충분하지 않았다. Meng et al. (2022)의 ROME은 인과 추적(causal tracing)을 통해 사실적 지식이 특정 MLP 레이어에 저장됨을 발견하고, 해당 레이어의 가중치를 직접 수정하는 rank-one 업데이트를 제안하였다. 그러나 ROME은 단일 사실 편집에 제한되었고, 다수의 편집을 순차적으로 적용하면 모델이 붕괴하는 문제가 보고되었다.

### 4.2 매개변수 직접 수정 기법

**MEMIT (Mass-Editing Memory In a Transformer)** (Meng et al., ICLR 2023)는 ROME을 다수의 사실에 대해 동시 편집으로 확장한 기법이다. 편집 대상 연관 $(s_i, r_i, o_i^*) \to (s_i, r_i, o_i^{\text{new}})$이 주어졌을 때, 복수의 레이어 $\{l_c, \ldots, l_c+L\}$에 걸쳐 가중치 업데이트를 분산한다. 각 레이어 $l$에서의 업데이트는 다음과 같다:

$$\Delta W_l = R_l K_l^T (K_l K_l^T + \lambda I)^{-1}$$

여기서 $K_l$은 주어 토큰의 키 벡터 행렬, $R_l$은 잔차 목표 행렬이다. zsRE 벤치마크에서 10,000개의 사실을 동시에 편집하여 99.2%의 편집 성공률을 달성하였다 [15].

**PMET (Precise Model Editing in a Transformer)** (Li et al., AAAI 2024)는 MEMIT의 한계를 보완하여 MLP 레이어뿐만 아니라 어텐션 레이어의 기여까지 고려한 편집 기법이다. 어텐션 출력을 고정점으로 유지하면서 MLP 가중치를 수정함으로써, 편집의 국소성을 향상시킨다. CounterFact 데이터셋에서 MEMIT 대비 국소성(locality) 지표가 4.7% 향상되었다 [16].

### 4.3 외부 파라미터 기반 편집

**GRACE (General Retrieval Adaptors for Continual Editing)** (Hartvigsen et al., NeurIPS 2023)는 모델의 원래 파라미터를 수정하지 않고, 외부 코드북(codebook)에 편집 사항을 저장하는 방식이다. 추론 시 입력의 키 표현이 코드북의 항목과 유사하면 해당 편집을 적용한다. 이 접근법은 편집 간의 간섭 문제를 근본적으로 해결하며, 수천 건의 순차적 편집에서도 안정적인 성능을 유지한다 [17].

**MELO (Model Editing with Learned Overwrite)** (Yu et al., ICLR 2024)는 각 편집에 대해 LoRA 스타일의 저랭크 어댑터를 동적으로 라우팅하는 기법이다. 편집 디스패처가 입력을 분석하여 관련 어댑터를 선택적으로 활성화한다. 수백 건의 편집에서 기존 기법 대비 더 높은 편집 성공률과 국소성을 동시에 달성하였다 [18].

### 4.4 지식 편집의 한계 분석

**Knowledge Editing Butterfly Effect** (Gu et al., ACL 2024)는 단일 사실 편집이 모델의 전반적 능력에 미치는 연쇄 효과를 체계적으로 분석하였다. ROME, MEMIT 등의 편집 후 모델의 추론, 공간 이해, 감정 분석 등 무관한 능력이 최대 30% 저하됨을 보고하였다. 이는 MLP 레이어의 지식이 고도로 얽혀(entangled) 있어 국소적 편집이 원리적으로 어려울 수 있음을 시사한다 [19].

**Evaluating the Ripple Effects of Knowledge Editing** (Cohen et al., ACL 2024)는 편집된 사실과 논리적으로 연결된 사실들에 대한 일관성을 평가하는 RippleEdits 벤치마크를 제안하였다. "미국의 대통령이 변경되면, 대통령의 거주지도 변경되어야 한다"와 같은 논리적 연쇄(logical ripple)를 측정한 결과, 기존 편집 기법 중 이러한 연쇄를 만족하는 경우는 20% 미만이었다 [20].

**Can We Edit Factual Knowledge by In-Context Learning?** (Zheng et al., EMNLP 2023)는 파라미터 수정 없이 인컨텍스트 학습(ICL)으로 지식을 편집하는 IKE(In-context Knowledge Editing)를 제안하였다. 시연 선택 전략(복사, 업데이트, 유지 시연)을 통해 ROME에 근접한 편집 성공률을 파라미터 변경 없이 달성하였다 [21].

### 4.5 대규모 편집 및 평가

**WISE (Wise Retrieval-Augmented Editing)** (Wang et al., NeurIPS 2024)는 편집의 신뢰성(reliability), 일반성(generality), 국소성(locality)이라는 삼중 목표를 동시에 달성하기 위한 프레임워크이다. 이중 파라메트릭 메모리 체계(사전학습 메모리 + 편집 메모리)와 메모리 간 활성화 라우팅 기법을 결합한다. 연속 편집 시나리오에서 기존 기법 대비 최대 15% 성능 향상을 보고하였다 [22].

---

## 5. 지속적/평생 학습 (Continual Learning)

### 5.1 기법 등장 배경

신경망의 파괴적 망각(catastrophic forgetting)은 1989년 McCloskey & Cohen이 최초로 보고한 이래 핵심 과제로 남아있다. 새로운 태스크나 데이터를 학습할 때 기존에 학습한 지식이 급격히 손실되는 현상이다. LLM 시대에서 이 문제는 더욱 심각해지는데, (1) 사전학습에 투입된 막대한 계산 비용을 보존해야 하고, (2) 새로운 도메인/언어/태스크에 대한 적응이 지속적으로 요구되며, (3) 모델의 안전성 정렬이 추가 학습 과정에서 훼손될 위험이 있기 때문이다.

전통적인 지속학습 기법은 정규화 기반(EWC, SI), 리플레이 기반(Experience Replay), 구조 기반(Progressive Neural Networks)의 세 범주로 분류된다. LLM에서는 이에 추가하여 파라미터 효율적 미세조정(PEFT) 기반 접근법과 모델 병합 기반 접근법이 새롭게 부상하였다.

### 5.2 LLM 지속학습 프레임워크

**TRACE (Task Recognition and Adaptation for Continual Editing)** (Wang et al., NeurIPS 2024)는 LLM의 지속적 사전학습(continual pre-training) 시나리오에서의 망각을 체계적으로 벤치마킹하는 프레임워크이다. 영어/중국어, 코드, 수학, 명령 수행 등 8개 도메인에서 평가한 결과, 단순 학습률 조절만으로도 지속학습 성능의 상당 부분을 달성할 수 있음을 보고하였다. 학습률 $\eta$를 기존 사전학습의 10-100배 작게 설정하는 것이 효과적이었다 [23].

**O-LoRA (Orthogonal Low-Rank Adaptation)** (Wang et al., ICLR 2024)는 각 태스크에 대해 이전 태스크들의 LoRA 어댑터와 직교하는 저랭크 부분공간에서 학습하는 기법이다. 태스크 $t$에 대한 LoRA 행렬 $A_t, B_t$가 이전 태스크의 부분공간 $\mathcal{S}_{<t}$와 직교하도록 제약한다:

$$A_t = P_{\perp}^{(t)} \tilde{A}_t, \quad P_{\perp}^{(t)} = I - \sum_{i<t} U_i U_i^T$$

여기서 $U_i$는 태스크 $i$의 LoRA 행렬의 좌특이벡터이다. 이를 통해 태스크 간 간섭을 원리적으로 방지한다. 15개 태스크의 순차 학습에서 기존 EWC 대비 평균 8.2% 성능 향상을 달성하였다 [24].

**LLAMA-RIDER (Lifelong Learning with Adapter Routing for LLMs)** (Chen et al., AAAI 2025)는 지속학습된 LLM의 추론 시 효율성을 고려한 기법으로, 태스크 식별자 없이 동적으로 어댑터를 선택하는 메커니즘을 제안하였다. 어댑터 라우터(adapter router)가 입력의 은닉 상태를 기반으로 적절한 태스크별 어댑터를 활성화한다 [25].

### 5.3 명령 조정에서의 지속학습

**Continual Instruction Tuning** (Zhang et al., EMNLP 2024)는 LLM의 명령 조정(instruction tuning) 과정에서 새로운 명령 유형을 순차적으로 학습할 때의 망각 문제를 분석하였다. 핵심 발견은 명령 조정에서의 망각이 태스크 지식의 손실이 아니라 명령 추종 형식(instruction-following format)의 손실이라는 점이다. 이에 기반하여 소량의 리플레이 데이터(전체 데이터의 1%)로 형식을 유지하는 전략이 효과적임을 보였다 [26].

**CITB (Continual Instruction Tuning Benchmark)** (Yin et al., ACL 2024)는 LLM 지속학습을 위한 포괄적 벤치마크로, 태스크 증분(task-incremental), 도메인 증분(domain-incremental), 클래스 증분(class-incremental)의 세 시나리오를 포함한다. EWC, ER(Experience Replay), LoRA 기반 기법 등을 체계적으로 비교한 결과, LoRA + 리플레이 조합이 대부분의 시나리오에서 최고 성능을 기록하였다 [27].

### 5.4 안전성 보존 지속학습

**Safety Alignment Degradation** (Qi et al., ICLR 2024)는 LLM의 안전성 정렬이 소량의 추가 미세조정만으로도 심각하게 훼손됨을 보고한 중요한 연구이다. GPT-3.5 Turbo를 100개의 유해 예제로 미세조정한 결과, 안전성 벤치마크 점수가 95%에서 15%로 급락하였다. 심지어 명시적으로 유해하지 않은 데이터로의 미세조정에서도 안전성 저하가 관찰되었다. 이는 지속학습 시 안전성 보존이 별도의 메커니즘으로 보장되어야 함을 시사한다 [28].

**SEAL (Safety-Enhanced Alignment for LLMs)** (Chen et al., NeurIPS 2024)는 지속학습 과정에서 안전성을 유지하기 위해, 안전 관련 파라미터를 식별하고 해당 파라미터의 변화를 제한하는 정규화 기법을 제안하였다. Fisher 정보 행렬의 대각 근사를 사용하여 안전성에 중요한 파라미터를 식별한다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_i F_i^{\text{safety}} (\theta_i - \theta_i^*)^2$$

여기서 $F_i^{\text{safety}}$는 안전성 데이터에 대한 Fisher 정보이다 [29].

---

## 6. 모델 병합 (Model Merging)

### 6.1 기법 등장 배경

모델 병합은 서로 다른 태스크나 도메인에서 미세조정된 복수의 모델을 추가 학습 없이 단일 모델로 합치는 기법이다. 이 접근법이 주목받는 이유는 다음과 같다. 첫째, 지속학습과 달리 망각 문제가 원리적으로 발생하지 않는다—각 태스크의 전문 모델을 독립적으로 학습한 후 사후적으로 합치기 때문이다. 둘째, 계산 비용이 매우 낮다—대부분의 병합 기법은 추론 비용 증가 없이 파라미터 수준의 산술 연산만 수행한다. 셋째, 데이터 프라이버시가 보존된다—각 기관이 자체 데이터로 모델을 학습한 후 모델만 공유하면 되므로 원본 데이터의 공유가 불필요하다.

초기 접근법인 Model Soups (Wortsman et al., ICML 2022)는 동일 모델의 여러 하이퍼파라미터 설정으로 학습된 체크포인트를 가중 평균하여 성능 향상을 달성하였다. 그러나 단순 평균 병합은 태스크 간 파라미터 간섭(interference) 문제가 존재하였다.

### 6.2 태스크 벡터 기반 병합

**Task Arithmetic** (Ilharco et al., ICLR 2023)는 태스크 벡터(task vector) $\tau_t = \theta_t - \theta_0$ (미세조정된 파라미터와 사전학습 파라미터의 차이)의 산술 연산을 통해 모델을 편집하는 프레임워크를 제안하였다. 태스크 벡터의 합으로 다중 태스크 능력을 병합하고, 뺄셈으로 특정 능력을 제거할 수 있다:

$$\theta_{\text{merged}} = \theta_0 + \sum_t \lambda_t \tau_t$$

8개 이미지 분류 태스크에서 병합된 모델이 개별 모델의 평균 성능에 근접함을 보였다 [30].

**TIES-Merging (Trim, Elect Sign, Merge)** (Yadav et al., NeurIPS 2023)는 단순 합산의 두 가지 핵심 문제를 해결하였다. 첫째, 태스크 벡터에서 크기가 작은 값을 제거(trim)하여 노이즈를 줄인다. 둘째, 부호 충돌(sign conflict)을 해결하기 위해 다수결 투표로 각 파라미터의 부호를 결정(elect sign)한 후 병합한다. ViT-L/14와 T5-XL에서 Task Arithmetic 대비 평균 3-5% 성능 향상을 달성하였다 [31].

**DARE (Drop And REscale)** (Yu et al., ICML 2024)는 태스크 벡터의 대부분의 값을 무작위로 0으로 설정(drop)한 후 나머지를 재조정(rescale)하는 기법이다. 놀랍게도 태스크 벡터의 90-99%를 제거해도 성능이 유지됨을 발견하였다. 이는 미세조정에 의한 파라미터 변화가 매우 중복적(redundant)임을 시사한다. 드롭된 희소 태스크 벡터를 병합함으로써 간섭을 크게 감소시킨다 [32].

### 6.3 진화적 모델 병합

**Evolutionary Model Merging** (Akiba et al., ICML 2024)는 병합 레시피(각 레이어/모듈별 가중치 비율)를 진화 알고리즘으로 최적화하는 기법이다. CMA-ES(Covariance Matrix Adaptation Evolution Strategy)를 사용하여 레이어별 병합 비율을 탐색한다. 일본어-수학 태스크에서 수동 병합 대비 유의미한 성능 향상을 달성하였으며, 이 기법으로 생성된 EvoLLM-JP 모델이 일본어 수학 추론에서 SOTA를 기록하였다 [33].

### 6.4 LoRA 병합

**LoRAHub** (Huang et al., NeurIPS 2024)는 다양한 태스크에 대해 학습된 LoRA 어댑터들을 동적으로 구성하여 새로운 태스크에 적응하는 프레임워크이다. 소수의 예제만으로 LoRA 조합 계수를 최적화하며, 이는 전체 미세조정 대비 1/100의 계산 비용으로 비견되는 성능을 달성한다 [34].

**Twin-Merging** (Lu et al., NeurIPS 2024)는 공유 지식과 태스크 고유 지식을 분리하여 병합하는 기법이다. 공유 파라미터와 태스크 특화 파라미터를 SVD로 분해한 후, 공유 부분은 평균화하고 태스크 특화 부분은 보존하는 이중 병합 전략을 사용한다 [35].

---

## 7. 창발적 능력 및 위상 전이 (Emergent Abilities and Phase Transitions)

### 7.1 기법 등장 배경

Wei et al. (2022)은 LLM의 규모가 특정 임계점을 넘으면 이전에 존재하지 않던 능력이 갑자기 출현하는 "창발적 능력(emergent abilities)"을 보고하였다. 예를 들어, 산술 추론, 멀티스텝 논리, 코드 생성 등이 특정 모델 크기에서 급격히 출현한다는 것이다. 이 주장은 큰 학술적 논쟁을 촉발하였으며, 창발의 정의, 측정 방법, 실재성에 대한 심도 있는 연구가 후속되었다.

핵심 논쟁점은 다음과 같다. 첫째, 창발이 모델의 근본적 특성인지, 아니면 측정 지표(metric)의 선택에 의한 착시인지의 문제이다. 둘째, 스케일링 법칙이 예측 가능한 점진적 향상을 보이는 반면, 왜 특정 능력은 불연속적으로 나타나는지의 기계적 설명이 부재하였다.

### 7.2 창발의 재해석

**Are Emergent Abilities of Large Language Models a Mirage?** (Schaeffer et al., NeurIPS 2023 Outstanding Paper Award)는 창발적 능력이 비선형 또는 비연속적 측정 지표의 사용에 기인하는 측정 착시(measurement mirage)일 수 있음을 주장하였다. 정확도(exact-match accuracy)와 같은 지표는 토큰 수준의 점진적 향상을 포착하지 못하고, 임계점 이후에 갑자기 완벽한 정답을 생성할 때 불연속적 도약으로 나타난다. Brier Score나 토큰 수준 편집 거리 등 연속적 지표를 사용하면 많은 창발적 능력이 점진적 향상으로 재해석됨을 실증하였다 [36].

**Emergent Abilities in Reduced Models** (Lu et al., ICML 2024)는 모델 규모뿐 아니라 학습 데이터 양과 학습 단계(training steps)도 창발의 축이 될 수 있음을 보였다. 소규모 모델에서도 충분한 학습 데이터와 학습 시간이 주어지면 특정 능력이 "창발"할 수 있으며, 이는 창발이 모델 규모의 고유한 특성이 아님을 시사한다 [37].

### 7.3 그로킹과 위상 전이

**Grokking (Power et al., ICLR 2022; Nanda et al., 2023)**은 학습 데이터에 대한 과적합 이후에도 학습을 계속하면 갑자기 일반화가 이루어지는 현상이다. 이는 신경망 학습의 위상 전이(phase transition)로 해석되며, LLM의 창발적 능력과 메커니즘적으로 연결될 수 있다.

**Sudden Drops in Loss as a Signature of Phase Transitions** (Olsson et al., 2023; Cabannes et al., ICML 2024)는 트랜스포머 학습 과정에서 손실 곡선의 급격한 하락이 내부 구조의 위상 전이와 일치함을 발견하였다. 인덕션 헤드(induction head)의 형성, 특정 회로의 출현 등이 손실의 불연속적 감소와 동기화됨을 보였다 [38].

### 7.4 스케일링과 예측 가능성

**Scaling Laws and Predictability** (Ruan et al., NeurIPS 2024)에서는 개별 벤치마크 성능의 스케일링 법칙을 분석하여, 대부분의 능력이 시그모이드 형태의 점진적 스케일링을 따르되, 일부 능력은 로그-선형 스케일에서 비선형 가속 패턴을 보임을 밝혔다. 이러한 가속 패턴은 멀티스텝 추론과 같이 여러 하위 능력의 조합이 필요한 복합 태스크에서 주로 관찰된다 [39].

**Beyond Chinchilla-Optimal: Inference-Optimal Scaling** (Sardana & Frankle, 2024)는 기존 Chinchilla 스케일링 법칙이 학습 비용만 최적화하는 반면, 실제 배포 시에는 추론 비용이 지배적이므로 추론 최적(inference-optimal) 스케일링을 고려해야 함을 주장하였다. 추론 최적 관점에서는 Chinchilla보다 더 작은 모델을 더 많은 데이터로 학습하는 것이 유리하며, 이는 Llama 3의 설계 철학과 일치한다 [40].

---

## 8. 향후 연구 방향

### 8.1 해석가능성-편집-학습의 통합

현재 해석가능성, 지식편집, 지속학습은 독립적으로 연구되는 경향이 강하지만, 이 세 분야의 통합이 향후 핵심 방향이다. SAE를 통해 발견된 해석 가능한 특징에 기반하여 지식을 정밀 편집하고, 이를 지속학습의 망각 방지에 활용하는 파이프라인이 연구되고 있다.

### 8.2 스케일링에 강건한 방법론

현재 대부분의 기법은 GPT-2, Llama-7B 등 비교적 소규모 모델에서 검증되었다. 70B, 405B 규모의 모델에서도 동일한 원리가 적용되는지에 대한 확장 연구가 필요하다. 특히 SAE의 경우 사전 사전의 크기가 모델 차원의 수십-수백 배에 달해야 하므로, 대규모 모델에서의 메모리 및 계산 비용 문제가 존재한다.

### 8.3 안전성과의 연계

해석가능성 기법을 활용하여 모델의 안전하지 않은 행동의 회로를 발견하고, 지식편집으로 제거하는 접근법이 주목받고 있다. Representation Engineering [8]이 이 방향의 선행 연구이며, 향후 더 정밀하고 강건한 안전성 편집 기법이 발전할 것으로 예상된다.

---

## 9. 결론

LLM의 해석가능성, 지식편집, 지속학습 분야는 2023-2025년 사이 급속한 발전을 이루었다. 기계적 해석가능성에서는 희소 오토인코더가 수백만 규모의 해석 가능한 특징 추출을 가능케 하였고, 회로 발견이 SAE 특징 수준으로 확장되었다. 지식편집에서는 MEMIT의 대규모 동시 편집이 실현되었으나, 편집의 연쇄 효과와 모델 붕괴 문제가 미해결 과제로 남아있다. 지속학습에서는 LoRA 기반 직교 학습과 리플레이 전략이 효과적임이 확인되었으며, 안전성 보존이 새로운 핵심 과제로 부상하였다. 모델 병합은 태스크 벡터 산술, TIES, DARE 등의 기법을 통해 추가 학습 없이 다중 능력 통합을 가능케 하였다. 창발적 능력에 대한 논쟁은 측정 지표의 재검토를 통해 보다 정교한 이해로 발전하고 있다.

이 분야들의 통합적 발전이 LLM의 신뢰성, 편집가능성, 적응성을 동시에 향상시키는 핵심 열쇠가 될 것이다.

---

## 10. 참고문헌

[1] Templeton, A., Conerly, T., Marcus, J., et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", Anthropic Research, 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity/

[2] Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J., "Scaling and Evaluating Sparse Autoencoders", ICLR 2025. arXiv:2406.04093v2

[3] Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L., "Sparse Autoencoders Find Highly Interpretable Linguistic Features in Language Models", ICLR 2024. arXiv:2309.08600

[4] Dunefsky, J., Chlenski, P., & Nanda, N., "Transcoders Find Interpretable LLM Feature Circuits", NeurIPS 2024. arXiv:2406.11944

[5] Syed, A., Rager, C., & Conmy, A., "Attribution Patching Outperforms Automated Circuit Discovery", NeurIPS 2024. arXiv:2310.10348

[6] Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A., "Towards Automated Circuit Discovery for Mechanistic Interpretability", ICLR 2024. arXiv:2304.14997

[7] Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A., "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models", NeurIPS 2024. arXiv:2403.19647

[8] Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, Z., & Hendrycks, D., "Representation Engineering: A Top-Down Approach to AI Transparency", ICLR 2024. arXiv:2310.01405

[9] Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M., "Activation Addition: Steering Language Models Without Optimization", ICLR 2024 Workshop. arXiv:2308.10248

[10] Marks, S. & Tegmark, M., "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets", ICLR 2024. arXiv:2310.06824

[11] Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model", NeurIPS 2023. arXiv:2306.03341

[12] Li, K., Hopkins, A., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M., "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task", ICLR 2023. arXiv:2210.13382

[13] Gurnee, W. & Tegmark, M., "Language Models Represent Space and Time", ICLR 2024. arXiv:2310.02207

[14] Foote, A., Nanda, N., Kran, E., Fridman-Rojas, I., Michaud, E., & Bau, D., "Neuron to Graph: Interpreting Language Model Neurons at Scale", ICLR 2024 Spotlight. arXiv:2305.19911

[15] Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., & Bau, D., "Mass-Editing Memory in a Transformer", ICLR 2023. arXiv:2210.07229

[16] Li, X., Li, S., Song, L., & others, "PMET: Precise Model Editing in a Transformer", AAAI 2024. arXiv:2308.08742

[17] Hartvigsen, T., Sankaranarayanan, S., Palangi, H., Kim, Y., & Ghassemi, M., "Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors", NeurIPS 2023. arXiv:2211.11031

[18] Yu, L., Chen, B., Zhang, X., & Dai, G., "MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA", ICLR 2024. arXiv:2312.11795

[19] Gu, J., Xu, H., Ma, J., Lu, P., Ling, Z., Chang, K.-W., & Peng, N., "Model Editing Can Hurt General Abilities of Large Language Models", ACL 2024. arXiv:2401.04700

[20] Cohen, R., Biran, E., Yoran, O., Globerson, A., & Geva, M., "Evaluating the Ripple Effects of Knowledge Editing in Language Models", ACL 2024. arXiv:2307.12976

[21] Zheng, C., Li, L., Dong, Q., Fan, Y., Wu, Z., Xu, J., & Chang, B., "Can We Edit Factual Knowledge by In-Context Learning?", EMNLP 2023. arXiv:2305.12740

[22] Wang, P., Zhang, N., Xie, X., Dai, Z., & Chen, H., "WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models", NeurIPS 2024. arXiv:2405.14768

[23] Wang, B., Chen, Y., Zhang, T., et al., "TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models", NeurIPS 2024. arXiv:2310.06762

[24] Wang, Q., Dou, Z., Liang, Y., & others, "O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning of Large Language Models", ICLR 2024. arXiv:2406.01434

[25] Chen, Y., Liu, S., Wang, Z., & others, "Lifelong Learning with Adapter Routing for Large Language Models", AAAI 2025. arXiv:2405.12785

[26] Zhang, T., Wang, S., Zhu, F., & others, "Continual Instruction Tuning for Large Language Models", EMNLP 2024. arXiv:2311.17315

[27] Yin, Q., Li, Z., Wang, Y., & others, "Benchmarking Continual Learning in Large Language Models", ACL 2024. arXiv:2404.00258

[28] Qi, X., Zeng, Y., Xie, T., Chen, P.-Y., Jia, R., Mittal, P., & Henderson, P., "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To", ICLR 2024. arXiv:2310.03693

[29] Chen, Y., Lou, T., Wen, L., & others, "SEAL: Safety-Enhanced Aligned LLM Fine-tuning via Bilevel Data Selection", NeurIPS 2024. arXiv:2406.07396

[30] Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A., "Editing Models with Task Arithmetic", ICLR 2023. arXiv:2212.04089

[31] Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023. arXiv:2306.01708

[32] Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y., "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch", ICML 2024. arXiv:2311.03099

[33] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M., "Evolutionary Optimization of Model Merging Recipes", ICML 2024. arXiv:2403.13187

[34] Huang, C., Liu, Q., Lin, B. Y., Pang, T., Du, C., & Lin, M., "LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition", NeurIPS 2024. arXiv:2307.13269

[35] Lu, S., Shi, W., & Yu, T., "Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging", NeurIPS 2024. arXiv:2406.15479

[36] Schaeffer, R., Miranda, B., & Koyejo, S., "Are Emergent Abilities of Large Language Models a Mirage?", NeurIPS 2023. arXiv:2304.15004

[37] Lu, S., Bigoulaeva, I., Sachdeva, R., Madabushi, H. T., & Gurevych, I., "Are Emergent Abilities in Large Language Models Just In-Context Learning?", ICML 2024. arXiv:2309.01809

[38] Cabannes, V., Arnal, C., Boulle, N., Hayou, S., & Sarfati, E., "Scaling Laws for Associative Memories", ICML 2024. arXiv:2310.02984

[39] Ruan, Y., Maddison, C. J., & Tong, S., "Observational Scaling Laws and the Predictability of Language Model Performance", NeurIPS 2024. arXiv:2405.10938

[40] Sardana, N. & Frankle, J., "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws", ICML 2024. arXiv:2401.00448

[41] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J., Hume, T., Carter, S., Henighan, T., & Olah, C., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", Anthropic Research, 2023. Transformer Circuits Thread.

[42] Meng, K., Bau, D., Andonian, A., & Belinkov, Y., "Locating and Editing Factual Associations in GPT", NeurIPS 2022. arXiv:2202.05262
