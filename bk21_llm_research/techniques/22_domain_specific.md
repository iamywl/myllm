# LLM 도메인 특화 및 다국어 연구 최신 동향 (2023-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI, KDD) 중심
> 총 참고논문: 42편

---

## 1. 개요

대규모 언어 모델(LLM)은 범용 자연어 처리에서 뛰어난 성능을 보이지만, 특정 도메인에 적용할 때 전문 지식의 부족, 도메인 특화 용어 처리 한계, 다국어 환경에서의 성능 저하 등 다양한 문제가 발생한다. 과학 연구, 의료, 법률, 금융, 교육 등 각 도메인은 고유한 데이터 분포, 전문 용어, 추론 패턴을 가지며, 범용 LLM으로는 이러한 도메인 고유의 요구사항을 충족하기 어렵다.

도메인 특화 LLM 연구는 크게 세 갈래로 진행되고 있다. 첫째, 과학 발견(drug discovery, protein engineering, materials science)을 위한 LLM으로, 분자 구조를 토큰화하고 과학적 추론을 수행하는 모델이 개발되고 있다. 둘째, 의료, 법률, 금융 등 사회적 도메인에 특화된 LLM으로, 도메인 코퍼스 기반 사전학습과 전문가 피드백 정렬이 핵심이다. 셋째, 다국어 및 교차언어 LLM으로, 저자원 언어(low-resource language)에서의 성능 격차 해소가 주요 과제이다.

본 문서에서는 (1) 과학 발견을 위한 LLM, (2) 수학적 추론 및 정리 증명, (3) 의료/임상 LLM, (4) 법률/금융 LLM, (5) 다국어 및 교차언어 LLM, (6) 교육 도메인 LLM, (7) 구조화 데이터 처리의 7개 분야로 나누어 2023-2025년 최신 연구를 정리한다.

---

## 2. 과학 발견을 위한 LLM (LLM for Scientific Discovery)

### 2.1 기법 등장 배경

전통적인 과학 연구에서 약물 발견, 단백질 설계, 신소재 탐색은 실험적 시행착오에 의존하여 수년에서 수십 년의 시간이 소요되었다. 기계학습의 도입으로 가상 스크리닝(virtual screening)과 분자 성질 예측이 가능해졌으나, 초기 모델들은 분자 표현의 한계(고정 크기 fingerprint, 제한된 그래프 신경망 등)로 인해 일반화 능력이 부족하였다.

LLM의 등장으로 분자를 SMILES(Simplified Molecular Input Line Entry System) 문자열이나 SELFIES로 토큰화하여 자연어와 동일한 방식으로 처리할 수 있게 되었다. 그러나 분자 언어와 자연어 사이의 근본적인 의미론적 차이(molecular semantics vs. natural language semantics)로 인해 단순 전이학습은 한계가 있었다. 이에 따라 분자 구조 인코더와 언어 모델을 정렬(alignment)하는 다중모달 접근법, 과학 논문 코퍼스에 대한 도메인 사전학습, 그리고 과학적 추론을 위한 특화 프롬프팅 기법이 발전하였다.

### 2.2 분자 및 약물 발견 LLM

**3D-MoLM** (Li et al., NeurIPS 2024)은 3D 분자 구조와 언어 모델을 정렬하는 최초의 체계적 프레임워크이다. 3D 분자-텍스트 프로젝터를 통해 분자의 3차원 구조 정보를 LLM의 입력 공간으로 매핑하며, 3D 분자-중심 명령 조정 데이터셋(3D-MoIT)을 구축하여 분자 성질 예측, 분자 캡셔닝, 텍스트 기반 분자 검색을 통합적으로 수행한다. 분자 성질 예측 태스크에서 기존 2D 기반 방법 대비 평균 12.8% 성능 향상을 달성하였다 [1].

**MoleculeSTM** (Liu et al., ICML 2023)은 화학 구조와 텍스트 설명을 대조 학습(contrastive learning)으로 정렬하는 다중모달 분자 모델이다. 280,012개 분자-텍스트 쌍으로 학습하여, 제로샷 분자-텍스트 검색에서 기존 방법 대비 AUC-ROC 0.87을 달성하였다. 텍스트 기반 분자 편집(text-based molecule editing)이라는 새로운 태스크를 정의하여, 자연어 명령으로 분자의 특정 성질을 조절하는 것이 가능함을 보였다 [2].

**DrugAssist** (Ye et al., KDD 2024 Workshop)는 대화형 약물 분자 최적화를 위한 LLM 기반 프레임워크이다. 약물 후보 분자의 ADMET(Absorption, Distribution, Metabolism, Excretion, Toxicity) 성질을 자연어 대화를 통해 반복적으로 최적화한다. 분자 생성 시 화학적 유효성(chemical validity)을 94.7%로 유지하면서 목표 성질 개선을 달성하였다 [3].

### 2.3 단백질 언어 모델

**ESM-2 및 ESMFold** (Lin et al., Science 2023, ICML 2023 관련 확장)는 150억 파라미터 규모의 단백질 언어 모델로, 아미노산 서열에 대한 마스크드 언어 모델링(masked language modeling)을 통해 학습된다. 학습된 표현으로부터 단일 서열만으로 원자 수준의 단백질 3D 구조를 예측하며, AlphaFold2와 비교하여 60배 빠른 추론 속도를 달성하였다. 단백질 접촉 예측에서 장거리 정밀도(long-range precision) L/5 = 0.84를 기록하였다 [4].

**ProteinChat** (Guo et al., ICML 2024 Workshop)은 단백질 구조 인코더와 LLM을 결합하여 단백질에 대한 자연어 질의응답을 수행하는 시스템이다. 단백질의 기능, 활성 부위(active site), 결합 파트너 등에 대한 질문에 과학적으로 정확한 답변을 생성하며, 기존 단백질 기능 예측 모델 대비 해석가능성을 크게 향상시켰다 [5].

### 2.4 재료과학 및 화학 LLM

**LLM-Prop** (Rubungo et al., AAAI 2024)는 결정 구조의 텍스트 설명으로부터 재료의 밴드갭(band gap), 형성 에너지(formation energy) 등 물리적 성질을 예측하는 LLM 기반 프레임워크이다. Materials Project 데이터베이스의 44,000개 결정 구조에 대해 학습하였으며, 밴드갭 예측에서 MAE 0.31 eV, 형성 에너지 예측에서 MAE 0.064 eV/atom을 달성하여 GNN 기반 방법(CGCNN, MEGNet)과 비교 가능한 성능을 보였다 [6].

**ChemCrow** (Bran et al., NeurIPS 2023 Workshop → ICML 2024)은 LLM 에이전트에 18개 전문 화학 도구(분자 검색, 반응 예측, 안전성 평가 등)를 통합한 시스템이다. GPT-4 기반 에이전트가 도구를 자율적으로 선택하여 유기합성 경로 계획, 약물 발견, 재료 설계 태스크를 수행한다. 전문 화학자 평가에서 자율적 화학 연구 보조 도구로서의 가능성을 확인하였으며, 합성 계획 태스크에서 기존 retrosynthesis 도구를 적절히 활용하여 84%의 실행 가능한 경로를 제안하였다 [7].

**SciGLM** (Zhang et al., ACL 2024 Findings)은 과학 분야(물리, 화학, 수학) 추론 능력을 강화한 LLM이다. 자기 반성 기반 명령 어노테이션(self-reflective instruction annotation) 프레임워크를 통해 대학 수준 과학 문제에 대한 단계별 추론 데이터를 자동 생성한다. SciEval 벤치마크에서 ChatGLM 대비 물리 +4.87%, 화학 +3.21%, 수학 +5.62%의 정확도 향상을 달성하였다 [8].

---

## 3. 수학적 추론 및 정리 증명 (Mathematical Reasoning and Theorem Proving)

### 3.1 기법 등장 배경

수학적 추론은 LLM의 핵심 능력 중 하나로 주목받고 있으나, 기존 LLM은 다단계 산술 연산 오류, 논리적 비약, 그리고 형식적 증명(formal proof) 생성 실패 등의 한계를 보였다. 초기 접근법인 Chain-of-Thought(CoT) 프롬프팅은 자연어 추론 체인을 통해 일정 수준의 개선을 달성하였으나, 복잡한 수학 문제에서의 정확성 보장이 어려웠다.

이러한 한계를 극복하기 위해 두 가지 주요 방향이 발전하였다. 첫째, 수학 특화 사전학습 및 미세조정을 통해 수학적 패턴 인식 능력을 강화하는 접근법이다. 둘째, Lean4, Isabelle 등 형식 검증 시스템과 LLM을 결합하여 기계 검증 가능한 증명을 생성하는 접근법이다. 최근에는 강화학습 기반 추론 학습(예: DeepSeek-R1의 GRPO)이 수학 추론 성능을 비약적으로 향상시키고 있다.

### 3.2 수학 특화 LLM

**DeepSeekMath** (Shao et al., NeurIPS 2024 Spotlight)는 수학 추론에 특화된 7B 파라미터 모델이다. 1200억 토큰의 수학 관련 웹 데이터를 Common Crawl에서 추출하여 사전학습하였으며, 새로운 강화학습 알고리즘 GRPO(Group Relative Policy Optimization)를 제안하였다. GRPO는 별도의 보상 모델 없이 그룹 내 상대적 보상을 활용하여 정책을 최적화한다:

$$L_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) \right]$$

여기서 $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$는 그룹 내 정규화된 어드밴티지이다. MATH 벤치마크에서 51.7% 정확도를 달성하여 Gemini-Ultra(53.2%)와 GPT-4(52.9%)에 근접한 성능을 7B 모델로 달성하였다 [9].

**InternLM-Math** (Ying et al., ICLR 2024 → arXiv 2024)는 통합 수학 추론 프레임워크로, 체인오브소트 추론, 보상 모델링, 형식 추론, 데이터 증강, 코드 인터프리터를 하나의 모델에 통합한다. MATH 벤치마크에서 InternLM2-Math-Plus-20B가 53.6%를 달성하였으며, Lean4 형식 증명에서 miniF2F 벤치마크 48.8%의 정확도를 기록하였다 [10].

**Llemma** (Azerbayev et al., ICLR 2024)는 Proof-Pile-2라는 550억 토큰 규모의 수학 특화 코퍼스에 대해 Code Llama를 계속 사전학습한 모델이다. GSM8K에서 Llemma-34B가 72.5%, MATH에서 25.0%의 정확도를 기록하였으며, 형식 증명 생성에서 Lean4와 Isabelle 모두에 대해 few-shot으로 유의미한 성능을 보였다. 학습 데이터와 코드를 완전히 공개하여 재현성을 보장하였다 [11].

### 3.3 형식적 정리 증명 (Formal Theorem Proving)

**LEGO-Prover** (Xin et al., ICLR 2024)는 Lean4에서 성장하는 기술 라이브러리(skill library)를 활용하는 LLM 기반 정리 증명 시스템이다. 증명 과정에서 유용한 보조 정리(lemma)를 자동으로 추출하고 라이브러리에 추가하여, 이후 증명에서 재사용한다. miniF2F 벤치마크에서 57.0%의 증명 성공률을 달성하여 기존 최고 성능(ReProver 47.0%) 대비 10.0%p 향상을 이루었다 [12].

**DeepSeek-Prover-V1.5** (Xin et al., NeurIPS 2024)는 Lean4 형식 증명을 위한 LLM으로, 전체 증명 생성(whole-proof generation)과 트리 탐색(tree search)을 결합하는 접근법을 제안하였다. 자연어 추론 체인을 Lean4 주석(comment)으로 삽입하여 형식 증명의 가독성과 성공률을 동시에 높인다. miniF2F-test에서 63.5%의 정확도를 달성하였다 [13].

**AlphaProof** (Google DeepMind, 2024)는 AlphaZero 스타일의 강화학습과 Gemini LLM을 결합하여 국제수학올림피아드(IMO) 수준의 문제를 형식적으로 증명하는 시스템이다. 2024년 IMO에서 6문제 중 4문제를 해결하여 은메달 수준(28/42점)의 성적을 달성하였다. 자연어 문제를 Lean4 형식 명제로 자동 변환하고, 몬테카를로 트리 탐색(MCTS)을 통해 증명 공간을 효율적으로 탐색한다 [14].

---

## 4. 의료 및 임상 LLM (Medical and Clinical LLM)

### 4.1 기법 등장 배경

의료 분야에서 LLM의 적용은 임상 의사결정 지원, 의학 문헌 분석, 환자-의사 대화 요약 등에서 큰 잠재력을 가진다. 그러나 범용 LLM은 의학 전문 용어에 대한 이해 부족, 의료 특화 추론 능력 결여, 그리고 환각으로 인한 오진 위험 등의 문제를 가진다. 초기 연구(PubMedBERT, BioGPT)는 의학 논문 코퍼스에 대한 도메인 적응 사전학습을 시도하였으나, 디코더 기반 LLM 시대에는 대화형 의학 AI에 대한 요구가 증가하였다.

이에 따라 대규모 의학 코퍼스 기반 사전학습, 의료 전문가 피드백 정렬, 의료 벤치마크(USMLE, MedQA) 기반 평가 등을 통합하는 의료 특화 LLM이 등장하였다. 특히 환자 프라이버시 보호와 의료 AI의 안전성이 핵심 과제로 부각되고 있다.

### 4.2 의료 특화 기반 모델

**Med-PaLM 2** (Singhal et al., Nature 2023, arXiv:2305.09617)는 Google의 의료 특화 LLM으로, USMLE 스타일의 MedQA 벤치마크에서 86.5%의 정확도를 달성하여 전문의 수준(expert physician level)에 도달하였다. 의료 전문가의 선호도 평가에서 기존 Med-PaLM 대비 장문 의료 질문 답변의 품질이 유의미하게 향상되었으며, 9개 평가 축(정확성, 근거 기반, 유해성 등) 중 8개에서 의사 답변과 동등하거나 우수한 평가를 받았다 [15].

**PMC-LLaMA** (Wu et al., AAAI 2024)는 480만 편의 PubMed Central 논문과 30,000개의 의학 교과서 데이터로 LLaMA를 계속 사전학습한 의료 특화 모델이다. 의료 명령 조정 데이터 202K로 미세조정하여 MedQA에서 LLaMA-2-7B 대비 +18.7%의 정확도 향상을 달성하였다. 의학 벤치마크 PubMedQA, MedMCQA, USMLE에서 일관된 성능 개선을 보였다 [16].

**BioMistral** (Labrak et al., ACL 2024)는 Mistral-7B를 PubMed Central 논문에 대해 계속 사전학습한 의료 특화 모델이다. 10개 의료 QA 벤치마크에서 평가하였으며, 양자화(quantization)와 병합(merge) 기법이 의료 도메인 성능에 미치는 영향을 체계적으로 분석하였다. MMLU Medical 서브셋에서 BioMistral-7B가 기존 BioGPT보다 우수한 성능을 달성하였다 [17].

### 4.3 임상 태스크 특화

**Clinical Camel** (Toma et al., AAAI 2024 Workshop)은 임상 시나리오에 특화된 대화형 의료 AI로, 100,000개의 합성 임상 대화를 GPT-4로 생성하여 미세조정하였다. 임상 추론 태스크(감별 진단, 치료 계획, 임상 노트 요약)에서 범용 LLM 대비 유의미한 성능 향상을 보였다 [18].

**MedAlign** (Fleming et al., EMNLP 2023)은 EHR(Electronic Health Record) 기반 임상 태스크에 대한 983개의 자연어 명령-EHR 쌍을 포함하는 벤치마크이다. 의사 7명이 작성한 고품질 명령을 포함하며, 임상 노트 요약, 약물 조정 추천, 퇴원 지침 생성 등 실제 임상 워크플로우를 반영한다. GPT-4가 최고 성능을 보였으나, 의사 평가 기준 54.8%의 태스크에서만 임상적으로 적절한 응답을 생성하여 개선의 여지가 크다는 점을 지적하였다 [19].

---

## 5. 법률 및 금융 LLM (Legal and Financial LLM)

### 5.1 기법 등장 배경

법률 분야에서는 판례 검색, 계약서 분석, 법률 자문 등에 LLM을 적용하려는 시도가 활발하다. 그러나 법률 용어의 특수성, 관할권별 법체계의 차이, 그리고 법률적 판단의 정확성에 대한 높은 요구로 인해 범용 LLM의 직접 적용은 한계가 있다. 특히 법률 환각(legal hallucination)은 존재하지 않는 판례를 인용하거나 잘못된 법률 조항을 제시하는 심각한 문제를 초래한다.

금융 분야 역시 시장 분석, 재무제표 분석, 감성 분석, 위험 평가 등에 LLM을 활용하려는 연구가 증가하고 있으나, 금융 데이터의 시계열적 특성, 수치 추론의 정확성, 실시간 정보 반영의 필요성 등이 고유한 과제로 존재한다.

### 5.2 법률 특화 LLM

**SaulLM-7B** (Colombo et al., AAAI 2024 → arXiv:2403.03883)는 300억 토큰 규모의 영미법 코퍼스(판례, 법률, 계약서)에 대해 Mistral-7B를 계속 사전학습한 법률 특화 모델이다. LegalBench 벤치마크(162개 태스크)에서 기존 범용 7B 모델 대비 평균 +6.2%의 정확도 향상을 달성하였다. 법률 특화 명령 조정을 위해 법률 전문가가 검증한 16,000개의 명령-응답 쌍을 구축하였다 [20].

**ChatLaw** (Cui et al., arXiv:2306.16092, ACL 2024 관련)은 중국 법률 분야에 특화된 LLM으로, 법률 조항 검색, 키워드 추출, 그리고 법률 추론을 통합하는 파이프라인을 제안하였다. 법률 환각 문제를 완화하기 위해 참조 데이터(reference data)에 기반한 답변 생성 메커니즘을 도입하여, 중국 사법시험(Chinese Bar Exam) 시뮬레이션에서 GPT-4 대비 우수한 성능을 보였다 [21].

### 5.3 금융 특화 LLM

**FinGPT** (Yang et al., NeurIPS 2023 Workshop → ACL 2024)은 오픈소스 금융 LLM 프레임워크로, 금융 뉴스, SEC 보고서, 소셜 미디어 데이터를 통합하여 금융 감성 분석, 주가 예측, 위험 평가를 수행한다. LoRA 기반 경량 미세조정을 통해 빈번한 모델 업데이트를 가능하게 하며, 금융 감성 분석에서 기존 FinBERT 대비 F1 점수 +4.3% 향상을 달성하였다 [22].

**BloombergGPT** (Wu et al., NeurIPS 2023 Workshop)는 Bloomberg이 구축한 3630억 토큰의 금융 특화 코퍼스(FinPile)와 3450억 토큰의 범용 코퍼스를 혼합하여 학습한 500억 파라미터 모델이다. 금융 감성 분석(FPB 데이터셋 F1: 0.751), 명명 개체 인식(NER), 금융 질의응답(FiQA)에서 GPT-3 대비 우수한 성능을 보이면서도 범용 벤치마크 성능을 유지하였다. 금융 데이터와 범용 데이터의 최적 혼합 비율 탐색이 핵심 기여이다 [23].

**DocFinQA** (Reddy et al., EMNLP 2024)는 긴 금융 문서(연간보고서, 10-K)에 대한 수치 추론 벤치마크로, 기존 FinQA 대비 문서 길이를 10배 이상 확장하였다. GPT-4의 정확도가 전체 문서 컨텍스트에서 43.4%로 하락하여, 장문서 금융 추론의 어려움을 정량적으로 입증하였다 [24].

---

## 6. 다국어 및 교차언어 LLM (Multilingual and Cross-lingual LLM)

### 6.1 기법 등장 배경

현재 LLM은 영어 중심으로 학습되어 있으며, 비영어권 언어, 특히 저자원 언어(low-resource language)에서 심각한 성능 저하를 보인다. GPT-4, LLaMA 등 주요 LLM의 학습 데이터에서 영어가 80% 이상을 차지하며, 이에 따라 비영어 언어에서의 추론 능력, 문화적 뉘앙스 이해, 안전성 정렬 등이 부족하다.

이러한 문제를 해결하기 위해 세 가지 접근법이 발전하였다. 첫째, 다국어 사전학습 데이터 확충 및 균형 있는 데이터 혼합(data mixing) 전략이다. 둘째, 영어에서 학습된 지식과 능력을 비영어 언어로 전이(transfer)하는 교차언어 전이학습이다. 셋째, 특정 언어에 대한 계속 사전학습(continual pre-training) 및 어휘 확장(vocabulary extension)을 통한 언어별 적응이다.

### 6.2 다국어 기반 모델

**Aya Model** (Ustun et al., ACL 2024)은 101개 언어를 지원하는 다국어 명령 조정 모델이다. Cohere의 Aya 프로젝트에서 수집된 인간 작성 다국어 명령 데이터셋과 합성 데이터를 결합하여 mT5-XXL(13B)을 미세조정하였다. 저자원 언어 포함 다국어 벤치마크에서 기존 다국어 모델(BLOOMZ, mT0) 대비 평균 +14.2% 승률(win rate)을 달성하였다. 특히 터키어, 아랍어, 힌디어 등 중자원 언어에서의 성능 향상이 두드러졌다 [25].

**Aya-23** (Aryabumi et al., ACL 2024)는 Aya 프로젝트의 후속 모델로, 23개 언어에 집중하여 Cohere의 Command R+ 아키텍처에 기반한 8B 및 35B 모델을 제공한다. 다국어 MMLU, 번역, 요약 태스크에서 Gemma, Mistral 등 동급 모델 대비 다국어 성능에서 우수한 결과를 보였다 [26].

**BLOOM** (BigScience Workshop, ACL 2023)은 1,000명 이상의 연구자가 참여한 대규모 다국어 LLM 프로젝트로, 46개 자연어와 13개 프로그래밍 언어로 학습된 1760억 파라미터 모델이다. ROOTS 코퍼스(1.6TB)에 대해 학습하였으며, 다국어 코퍼스의 데이터 거버넌스(data governance)를 체계적으로 수립한 최초의 대규모 프로젝트이다. 다국어 생성 품질에서 언어별 성능 편차가 학습 데이터 비율과 높은 상관관계(r=0.87)를 가진다는 분석 결과를 제시하였다 [27].

### 6.3 저자원 언어 적응 및 교차언어 전이

**GlotLID** (Kargaran et al., EMNLP 2024)는 2,000개 이상 언어를 지원하는 언어 식별(language identification) 모델이다. 기존 LID 도구(fastText, CLD3)가 저자원 언어를 정확히 식별하지 못하는 문제를 해결하여, 웹 크롤링 데이터에서 저자원 언어 코퍼스를 정확히 추출할 수 있게 한다. 이는 저자원 언어 LLM 학습의 핵심 전처리 단계로 기능한다 [28].

**Cross-Lingual Transfer with Target Language Adapters** (Parovic et al., EMNLP 2023)는 소스 언어(주로 영어)에서 학습된 태스크 지식을 저자원 타겟 언어로 전이하는 어댑터 기반 방법론을 제안하였다. 언어 어댑터와 태스크 어댑터를 분리하여 조합하는 MAD-X 프레임워크를 확장하며, 31개 언어에 대한 실험에서 전체 모델 미세조정 대비 파라미터 효율성을 7.5배 향상시키면서 비교 가능한 성능을 유지하였다 [29].

**SERENGETI** (Adebara et al., ACL 2024)는 517개 아프리카 언어와 방언을 포함하는 다국어 LLM이다. 아프리카 언어에 대한 학습 데이터 부족 문제를 해결하기 위해, 웹 크롤링, 종교 텍스트, 정부 문서 등 다양한 소스에서 데이터를 수집하였다. AfriSenti(감성 분석), MasakhaNER(개체명 인식), AfriQA(질의응답) 벤치마크에서 기존 다국어 모델(mBERT, XLM-R) 대비 평균 +8.3%의 F1 점수 향상을 달성하였다 [30].

**MaLA-500** (Lin et al., ACL 2024)은 534개 언어를 커버하는 LLM으로, LLaMA-2를 Glot500-c 코퍼스에 대해 계속 사전학습하여 구축하였다. 기존 다국어 모델이 100개 미만의 언어만 지원하는 한계를 극복하고, SIB-200 벤치마크(topic classification, 204개 언어)에서 BLOOM-7B 대비 평균 +11.2%의 정확도 향상을 달성하였다 [31].

### 6.4 다국어 토크나이제이션 및 효율성

다국어 LLM의 핵심 과제 중 하나는 토크나이저의 언어별 효율성 불균형이다. 영어 중심 BPE 토크나이저는 비라틴 스크립트 언어(한국어, 아랍어, 태국어 등)에서 토큰당 표현 가능한 정보량이 2-10배 적어, 동일 의미의 텍스트를 처리하는 데 더 많은 토큰이 필요하다. 이는 추론 비용 증가와 컨텍스트 길이 제약으로 이어진다.

**TokenMonster** (Garza, 2024)와 같은 최적화된 다국어 토크나이저 연구가 진행되고 있으며, Gemma, Qwen2 등 최신 모델은 어휘 크기를 256K 이상으로 확장하여 다국어 토큰 효율성을 개선하고 있다. Qwen2의 경우 151,643개 어휘를 사용하여 중국어 토큰 효율성을 LLaMA-2 대비 2.4배 향상시켰다 [32].

---

## 7. 교육 도메인 LLM (LLM for Education)

### 7.1 기법 등장 배경

교육 분야에서 LLM의 활용은 개인화된 튜터링, 자동 문항 생성, 에세이 평가, 학습자 피드백 제공 등에서 높은 잠재력을 가진다. 기존의 지능형 튜터링 시스템(ITS)은 규칙 기반(rule-based)으로 설계되어 확장성과 유연성이 제한적이었다. LLM의 등장으로 자연어 기반 대화형 튜터링이 가능해졌으나, 교육적 효과성 검증, 잘못된 정보 전달 위험, 학습자 수준에 맞는 적응적 설명 생성 등이 과제로 남아 있다.

### 7.2 LLM 기반 튜터링 및 평가

**Khanmigo 및 GPT-4 기반 교육 연구** (OpenAI, 2023-2024)에서 Khan Academy는 GPT-4를 활용한 소크라테스식 튜터링 시스템을 개발하였다. 직접 답을 제공하지 않고 안내 질문(guiding questions)을 통해 학습자의 사고를 유도하는 방식으로, 무작위 대조 시험(RCT)에서 수학 성취도에 유의미한 향상을 보였다.

**EduChat** (Dan et al., ACL 2024 Findings)은 교육에 특화된 대화형 AI로, 교육 심리학 원리(scaffolding, zone of proximal development)를 프롬프트 설계에 반영하였다. 3단계 학습 파이프라인(사전학습 → 교육 코퍼스 미세조정 → RLHF)을 통해 교육적 대화 품질을 향상시켰다 [33].

**Automated Essay Scoring with LLMs** (Mizumoto et al., EMNLP 2023)는 GPT-4를 활용한 자동 에세이 채점 시스템의 가능성과 한계를 분석하였다. TOEFL 에세이 데이터셋에서 GPT-4의 채점이 인간 채점자와 QWK(Quadratic Weighted Kappa) 0.71의 일치도를 보였으나, 세부 채점 기준(rubric)에 대한 정밀한 평가에서는 인간 전문가에 미치지 못하였다. 프롬프트 엔지니어링과 few-shot 예시 제공이 채점 일치도를 유의미하게 향상시킨다는 결과를 제시하였다 [34].

**MathDial** (Macina et al., EMNLP 2023)은 수학 튜터링 대화 데이터셋으로, 교사-학생 역할극 대화 3,000여 건을 수집하여 LLM 기반 수학 튜터의 학습 및 평가에 활용한다. 교육적으로 적절한 피드백(힌트 제공, 오개념 교정)과 교육적으로 부적절한 피드백(직접 답 제공)을 구분하는 자동 평가 메트릭을 제안하였다 [35].

---

## 8. 구조화 데이터 처리 (Structured Data Processing)

### 8.1 기법 등장 배경

기업과 조직의 데이터 대부분은 관계형 데이터베이스, 스프레드시트, 지식 그래프 등 구조화된 형태로 존재한다. 자연어로 이러한 구조화 데이터를 질의하고 분석하려는 수요가 증가하면서, Text-to-SQL, 테이블 질의응답(Table QA), 그래프 기반 추론 등의 연구가 활발해졌다. 기존 시맨틱 파싱(semantic parsing) 기반 접근법은 도메인별 문법 규칙 정의가 필요하여 확장성이 제한적이었다.

LLM의 등장으로 자연어 질문을 직접 SQL 쿼리나 구조화된 질의로 변환하는 것이 가능해졌으나, 스키마 이해, 복잡한 조인(join) 연산, 서브쿼리 생성, 그리고 실행 정확성(execution accuracy) 보장 등의 과제가 존재한다. 이에 따라 스키마 링킹(schema linking), 자기 수정(self-correction), 실행 기반 검증(execution-based verification) 등의 기법이 발전하였다.

### 8.2 Text-to-SQL

**DIN-SQL** (Pourreza & Rafiei, NeurIPS 2023)은 Text-to-SQL 태스크를 하위 문제로 분해(decompose)하는 프롬프팅 기법이다. 스키마 링킹, 쿼리 분류 및 분해, SQL 생성, 자기 수정의 4단계 파이프라인을 통해 GPT-4 기반으로 Spider 벤치마크에서 실행 정확도(execution accuracy) 85.3%를 달성하여, 당시 기존 최고 성능을 크게 상회하였다. 특히 자기 수정(self-correction) 모듈이 4.3%p의 성능 향상에 기여하였다 [36].

**DAIL-SQL** (Gao et al., AAAI 2024)은 Text-to-SQL에서 in-context learning 예시 선택 전략을 체계적으로 분석한 연구이다. 질문 유사도, SQL 유사도, 그리고 마스킹된 SQL 유사도를 결합한 DAIL 선택 전략을 제안하여, Spider 벤치마크에서 GPT-4 기반 86.6%의 실행 정확도를 달성하였다. 기존 미세조정 기반 방법과 비교 가능한 성능을 프롬프팅만으로 달성한 것이 핵심 기여이다 [37].

**MAC-SQL** (Wang et al., ACL 2024 Findings)은 다중 에이전트 협업(multi-agent collaboration) 기반 Text-to-SQL 프레임워크이다. 분해기(Decomposer), SQL 생성기(Generator), 검증기(Refiner)의 세 에이전트가 협력하여 복잡한 SQL 쿼리를 생성한다. BIRD 벤치마크에서 GPT-4 기반 59.59%의 실행 정확도를 달성하여, 단일 에이전트 대비 +4.81%p 향상을 이루었다 [38].

### 8.3 테이블 이해 및 질의응답

**TableLlama** (Zhang et al., AAAI 2024)는 테이블 이해에 특화된 오픈소스 LLM으로, 14개 테이블 태스크(테이블 QA, 사실 검증, 테이블-텍스트 생성 등)에 대한 통합 학습을 수행한다. TableInstruct라는 250만 샘플 규모의 명령 조정 데이터셋을 구축하였으며, 테이블 QA(WikiTableQuestions)에서 LLaMA-2-7B 대비 +22.3%의 정확도 향상을 달성하였다 [39].

**TAPAS 후속 연구 및 Large Table Reasoning** (Sui et al., EMNLP 2023)는 대규모 테이블(수백 행 이상)에 대한 LLM 추론의 한계를 분석하였다. 테이블 크기가 증가함에 따라 LLM 성능이 급격히 하락하는 현상을 정량화하고, 테이블 분할(table partition), 관련 행 필터링(row filtering), 열 선택(column selection) 등의 전처리 전략이 성능 저하를 완화함을 보였다 [40].

### 8.4 지식 그래프와 LLM

**Think-on-Graph** (Sun et al., ICLR 2024)은 LLM과 지식 그래프(knowledge graph)를 결합한 추론 프레임워크이다. LLM이 질의를 분석하여 지식 그래프의 관련 엔티티를 식별하고, 그래프 탐색을 통해 추론 경로를 구성한 후, 이를 기반으로 최종 답변을 생성한다. 그래프 구조가 LLM의 환각을 억제하는 앵커(anchor) 역할을 수행하며, KGQA 벤치마크(WebQuestionsSP, CWQ)에서 기존 방법 대비 정확도 +8.7% 향상을 달성하였다 [41].

**Graph Neural Prompting** (Tian et al., AAAI 2024)은 지식 그래프 정보를 LLM의 프롬프트로 변환하는 학습 가능한 프레임워크이다. GNN을 통해 지식 그래프의 구조적 정보를 인코딩하고, 이를 소프트 프롬프트(soft prompt)로 변환하여 LLM에 주입한다. CommonsenseQA에서 +3.4%, OpenbookQA에서 +5.1%의 정확도 향상을 달성하였다 [42].

---

## 9. 종합 분석 및 향후 연구 방향

### 9.1 도메인 특화 LLM의 공통 패턴

본 문서에서 분석한 도메인 특화 LLM 연구들은 다음과 같은 공통 패턴을 가진다.

첫째, **도메인 적응 전략**은 크게 (a) 도메인 코퍼스 계속 사전학습, (b) 도메인 특화 명령 조정, (c) 도메인 도구 통합의 세 가지로 분류된다. 과학(ChemCrow), 의료(PMC-LLaMA), 법률(SaulLM) 등 각 도메인에서 이 세 전략이 단독 또는 조합되어 활용된다.

둘째, **평가의 전문성**이 도메인별로 고도화되고 있다. USMLE(의료), LegalBench(법률), Spider/BIRD(SQL), miniF2F(수학) 등 도메인별 전문 벤치마크가 구축되어 있으며, 각 벤치마크는 해당 도메인 전문가의 검증을 거친다.

셋째, **안전성과 신뢰성**이 도메인 특화에서 더욱 중요하다. 의료 환각, 법률 환각, 금융 오류 등은 범용 NLP에서의 오류보다 훨씬 큰 사회적 영향을 가지며, 이에 따라 도메인별 안전성 평가 프레임워크의 필요성이 대두되고 있다.

### 9.2 향후 연구 방향

1. **다국어 도메인 특화의 교차점**: 현재 도메인 특화 LLM은 대부분 영어 중심이다. 비영어권 법률, 의료, 과학 데이터에 대한 도메인 특화 모델 개발이 필요하다.
2. **실세계 검증(Real-world Validation)**: 벤치마크 성능을 넘어 실제 임상 환경, 법률 실무, 과학 실험에서의 LLM 효과성 검증이 필요하다.
3. **도메인 간 지식 전이**: 의료-화학, 법률-금융 등 관련 도메인 간 지식을 효과적으로 공유하는 방법론이 부족하다.
4. **저자원 언어 + 저자원 도메인**: 저자원 언어에서의 도메인 특화 데이터 부족은 이중적 과제이며, 합성 데이터 생성과 교차언어 전이의 결합이 유망한 방향이다.

---

## 10. 참고문헌

[1] Li, S. et al., "3D-MoLM: Towards 3D Molecule-Text Interpretation in Language Models", NeurIPS 2024. arXiv:2401.13923

[2] Liu, S. et al., "Multi-modal Molecule Structure-text Model for Text Based Retrieval and Editing", ICML 2023. arXiv:2212.10789

[3] Ye, J. et al., "DrugAssist: A Large Language Model for Molecule Optimization", KDD 2024 Workshop. arXiv:2401.10334

[4] Lin, Z. et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023. (ESM-2/ESMFold)

[5] Guo, H. et al., "ProteinChat: Towards Enabling ChatGPT-Like Capabilities on Protein 3D Structures", ICML 2024 Workshop. arXiv:2402.09568

[6] Rubungo, A. N. et al., "LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions", AAAI 2024. arXiv:2310.14029

[7] Bran, A. M. et al., "ChemCrow: Augmenting large-language models with chemistry tools", ICML 2024. arXiv:2304.05376

[8] Zhang, D. et al., "SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning", ACL 2024 Findings. arXiv:2401.07950

[9] Shao, Z. et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", NeurIPS 2024 Spotlight. arXiv:2402.03300

[10] Ying, H. et al., "InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning", ICLR 2024. arXiv:2402.06332

[11] Azerbayev, Z. et al., "Llemma: An Open Language Model for Mathematics", ICLR 2024. arXiv:2310.10631

[12] Xin, H. et al., "LEGO-Prover: Neural Theorem Proving with Growing Libraries", ICLR 2024. arXiv:2310.00656

[13] Xin, H. et al., "DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search", NeurIPS 2024. arXiv:2408.08152

[14] AlphaProof Team, Google DeepMind, "AI achieves silver-medal standard solving International Mathematical Olympiad problems", 2024.

[15] Singhal, K. et al., "Towards Expert-Level Medical Question Answering with Large Language Models", Nature 2023. arXiv:2305.09617

[16] Wu, C. et al., "PMC-LLaMA: Towards Building Open-source Language Models for Medicine", AAAI 2024. arXiv:2304.14454

[17] Labrak, Y. et al., "BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains", ACL 2024. arXiv:2402.10373

[18] Toma, A. et al., "Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding", AAAI 2024 Workshop. arXiv:2305.12031

[19] Fleming, S. L. et al., "MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records", EMNLP 2023. arXiv:2308.14089

[20] Colombo, P. et al., "SaulLM-7B: A pioneering Large Language Model for Law", AAAI 2024. arXiv:2403.03883

[21] Cui, J. et al., "ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases", arXiv:2306.16092, 2023.

[22] Yang, H. et al., "FinGPT: Open-Source Financial Large Language Models", ACL 2024. arXiv:2306.06031

[23] Wu, S. et al., "BloombergGPT: A Large Language Model for Finance", NeurIPS 2023 Workshop. arXiv:2303.17564

[24] Reddy, S. et al., "DocFinQA: A Long-Context Financial Reasoning Dataset", EMNLP 2024. arXiv:2401.10286

[25] Ustun, A. et al., "Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model", ACL 2024. arXiv:2402.07827

[26] Aryabumi, V. et al., "Aya 23: Open Weight Releases to Further Multilingual Progress", ACL 2024. arXiv:2405.15032

[27] BigScience Workshop, "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model", ACL 2023. arXiv:2211.05100

[28] Kargaran, A. H. et al., "GlotLID: Language Identification for Low-Resource Languages", EMNLP 2024. arXiv:2310.16248

[29] Parovic, M. et al., "Cross-Lingual Transfer with Target Language Adapters", EMNLP 2023. arXiv:2305.15455

[30] Adebara, I. et al., "SERENGETI: Massively Multilingual Language Models for Africa", ACL 2024. arXiv:2401.14866

[31] Lin, P. et al., "MaLA-500: Massive Language Adaptation of Large Language Models", ACL 2024. arXiv:2401.13303

[32] Yang, A. et al., "Qwen2 Technical Report", arXiv:2407.10671, 2024.

[33] Dan, Y. et al., "EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education", ACL 2024 Findings. arXiv:2308.02773

[34] Mizumoto, T. et al., "Exploring the Use of Large Language Models for Automated Essay Scoring", EMNLP 2023.

[35] Macina, J. et al., "MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems", EMNLP 2023. arXiv:2305.14536

[36] Pourreza, M. & Rafiei, D., "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction", NeurIPS 2023. arXiv:2304.11015

[37] Gao, D. et al., "Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation", AAAI 2024. arXiv:2308.15363

[38] Wang, B. et al., "MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL", ACL 2024 Findings. arXiv:2312.11242

[39] Zhang, T. et al., "TableLlama: Towards Open Large Generalist Models for Tables", AAAI 2024. arXiv:2311.09206

[40] Sui, Y. et al., "TAP4LLM: Table Provider on Sampling, Augmenting, and Packing Semi-structured Data for Large Language Model Reasoning", EMNLP 2023. arXiv:2312.09039

[41] Sun, J. et al., "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph", ICLR 2024. arXiv:2307.07697

[42] Tian, Y. et al., "Graph Neural Prompting with Large Language Models", AAAI 2024. arXiv:2309.15427
