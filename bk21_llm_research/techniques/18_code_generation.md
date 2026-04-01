# LLM 코드 생성 및 소프트웨어 공학 최신 연구 동향 (2023-2025)

> BK21 우수학회 (NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI, ICSE, FSE, ASE) 중심
> 총 참고논문: 42편

---

## 1. 개요

대규모 언어 모델(LLM)의 코드 생성 및 소프트웨어 공학 적용은 2023-2025년 기간 동안 가장 급속히 발전한 연구 분야 중 하나이다. 초기의 코드 생성 모델은 단일 함수 수준의 자동 완성에 머물렀으나, 최근에는 레포지토리 수준의 코드 이해, 자동 디버깅, 테스트 생성, 그리고 소프트웨어 개발 에이전트까지 확장되었다.

코드 생성 분야의 핵심 도전 과제는 다음과 같이 정리된다. 첫째, 자연어 명세에서 정확한 프로그램을 합성하는 문제(program synthesis)이다. 둘째, 생성된 코드의 기능적 정확성을 보장하는 문제로, 이는 실행 기반 검증(execution-based verification)과 형식 검증(formal verification)을 필요로 한다. 셋째, 실제 소프트웨어 개발 워크플로우에서의 적용으로, 코드 리뷰, 버그 수정, 테스트 작성 등 다양한 하위 태스크를 포함한다. 넷째, 레포지토리 수준의 장문맥 코드 이해로, 수천 개의 파일과 복잡한 의존성 구조를 처리해야 한다.

본 문서에서는 (1) 코드 생성 기반 모델, (2) 코드 추론 및 프로그램 합성, (3) 자동 디버깅 및 버그 수정, (4) 코드 리뷰 및 품질 보증, (5) 레포지토리 수준 코드 이해, (6) 테스트 생성, (7) LLM 기반 소프트웨어 에이전트의 7개 분야로 나누어 최신 연구를 정리한다.

---

## 2. 코드 생성 기반 모델 (Code Generation Foundation Models)

### 2.1 기법 등장 배경

범용 LLM(GPT-3, LLaMA 등)은 자연어 처리에 최적화된 사전학습 데이터 구성을 가지므로, 코드 생성 태스크에서 최적의 성능을 발휘하지 못하는 한계가 존재하였다. 특히 프로그래밍 언어의 구문적 엄밀성, 실행 가능성, 다중 파일 간 의존성 등 자연어와 상이한 특성을 학습하기 위해서는 코드 특화 사전학습이 필수적이다. 이에 Codex(Chen et al., 2021)를 시작으로 코드 특화 LLM이 등장하였으며, 2023년 이후에는 오픈소스 코드 모델의 성능이 상용 모델에 근접하거나 이를 초월하는 수준까지 발전하였다. 주요 발전 방향은 (i) 대규모 코드 코퍼스 큐레이션, (ii) 채우기(fill-in-the-middle, FIM) 학습, (iii) 긴 문맥 지원, (iv) 명령어 튜닝(instruction tuning) 등이다.

### 2.2 CodeLlama: Open Foundation Models for Code

**CodeLlama** (Rozière et al., 2024)는 Meta에서 공개한 코드 특화 LLM 군(family)으로, LLaMA 2를 기반으로 500B 토큰의 코드 데이터로 추가 사전학습하였다. 7B, 13B, 34B, 70B의 네 가지 크기로 제공되며, CodeLlama-Python(Python 특화)과 CodeLlama-Instruct(명령어 튜닝) 변형을 포함한다. 핵심 기술 기여는 다음과 같다.

- **Long Context Fine-Tuning (LCFT)**: 16,384 토큰의 긴 문맥을 지원하기 위하여 RoPE(Rotary Position Embedding)의 주파수를 조정하는 기법을 적용하였다. 구체적으로, RoPE의 base frequency $\theta$를 $10^4$에서 $10^6$으로 증가시켜 최대 100K 토큰까지 안정적인 추론을 가능하게 하였다.
- **Fill-in-the-Middle (FIM)**: 사전학습 시 코드의 일부를 마스킹하고 이를 채우는 방식의 학습을 적용하여, 코드 삽입(infilling) 태스크를 지원한다. FIM 학습 비율은 전체의 약 50%이다.
- **성능**: HumanEval에서 CodeLlama-34B가 48.8% pass@1을, CodeLlama-70B-Instruct가 67.8% pass@1을 달성하였다. MBPP에서는 70B 모델이 69.6%를 기록하였다 [1].

### 2.3 StarCoder2

**StarCoder2** (Lozhkov et al., 2024)는 BigCode 프로젝트의 후속 모델로, The Stack v2 데이터셋(67.5TB, 619개 프로그래밍 언어)에 기반하여 학습되었다. 3B, 7B, 15B의 세 가지 크기로 공개되었으며, 핵심 특성은 다음과 같다.

- **데이터 큐레이션**: GitHub 이슈, Pull Request, Jupyter 노트북, 코드 문서 등 다양한 소스를 포함하며, 개인정보 보호(PII) 필터링과 중복 제거(deduplication)를 체계적으로 수행하였다.
- **Grouped Query Attention (GQA)**: 추론 효율성을 위하여 GQA를 적용하고, 16,384 토큰의 문맥 창을 지원한다.
- **성능**: StarCoder2-15B는 HumanEval에서 46.3% pass@1, MBPP에서 65.4%를 달성하여 동일 크기 대비 최고 성능을 기록하였다. 특히 StarCoder2-3B는 StarCoder1-15B와 유사한 성능을 달성함으로써 데이터 품질의 중요성을 입증하였다 [2].

### 2.4 DeepSeek-Coder 및 DeepSeek-Coder-V2

**DeepSeek-Coder** (Guo et al., 2024)는 2조(2T) 토큰의 코드와 자연어 데이터로 처음부터(from scratch) 학습된 코드 모델이다. 1.3B에서 33B까지의 크기로 제공된다.

- **레포지토리 수준 사전학습**: 파일 단위가 아닌 레포지토리 수준에서 코드를 조직화하여 학습함으로써, 파일 간 의존성과 프로젝트 구조를 학습할 수 있도록 하였다. 구체적으로, 동일 레포지토리의 파일들을 의존성 그래프에 따라 정렬(topological sort)하여 하나의 학습 시퀀스로 구성한다.
- **16K 문맥 창**: 학습 초기부터 16,384 토큰의 문맥 창을 사용하여 레포지토리 수준 태스크를 지원한다.
- **성능**: DeepSeek-Coder-33B-Instruct는 HumanEval에서 79.3% pass@1을 달성하여 GPT-3.5-Turbo(72.6%)를 상회하였다.

**DeepSeek-Coder-V2** (Zhu et al., 2024)는 Mixture of Experts(MoE) 아키텍처를 채택하여 236B 파라미터 중 21B만을 활성화하는 구조이다. 128K 토큰의 문맥 창을 지원하며, HumanEval에서 90.2% pass@1을 달성하여 GPT-4-Turbo(90.2%)와 동등한 성능을 보였다 [3].

### 2.5 Qwen2.5-Coder

**Qwen2.5-Coder** (Hui et al., 2024)는 5.5조(5.5T) 토큰의 코드 관련 데이터로 학습된 코드 특화 모델이다. 0.5B에서 32B까지 다양한 크기로 제공된다.

- **파일 수준 및 레포지토리 수준 사전학습**: 코드 데이터를 파일 단위와 레포지토리 단위로 혼합하여 학습하였으며, 고품질 합성 데이터(synthetic data)를 적극 활용하였다.
- **성능**: Qwen2.5-Coder-32B-Instruct는 HumanEval에서 92.7% pass@1, MBPP에서 90.2%를 달성하여, 공개 모델 중 최고 수준의 성능을 기록하였다. EvalPlus 벤치마크에서 GPT-4o와 동등한 성능을 보였다 [4].

### 2.6 OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

**OpenCoder** (Huang et al., ICLR 2025)는 코드 LLM의 학습 과정을 완전히 재현 가능하도록 투명하게 공개한 최초의 연구이다. 1.5B와 8B 크기의 모델을 제공하며, RefineCode라 명명된 정제 코드 코퍼스(960B 토큰, 607개 언어)를 구축하였다. 핵심 기여는 데이터 정제 파이프라인, 학습 하이퍼파라미터, 중간 체크포인트를 모두 공개하여 코드 LLM 학습의 재현성을 확보한 것이다. HumanEval에서 8B 모델이 79.9% pass@1을 달성하였다 [5].

---

## 3. 코드 추론 및 프로그램 합성 (Code Reasoning and Program Synthesis)

### 3.1 기법 등장 배경

초기 코드 생성 모델은 단순한 함수 수준의 코드를 자연어 설명으로부터 생성하는 데 집중하였으나, 알고리즘적 사고가 필요한 복잡한 문제에서는 성능이 급격히 저하되었다. 예를 들어, 경쟁 프로그래밍(competitive programming) 문제에서는 문제의 수학적 구조를 이해하고, 적절한 알고리즘을 선택하며, 이를 정확하게 구현해야 한다. 이러한 한계를 극복하기 위하여 Chain-of-Thought(CoT) 추론, 실행 기반 검증, 자기 수정(self-refinement), 그리고 형식 검증 등의 기법이 등장하였다. 특히 2024년 이후에는 추론 시간 연산(test-time compute) 확장을 통한 코드 추론 강화가 핵심 연구 방향으로 부상하였다.

### 3.2 실행 기반 검증과 자기 수정

**CodeT** (Chen et al., ICLR 2023)는 LLM이 생성한 코드 후보들에 대하여 동시에 테스트 케이스를 생성하고, 생성된 테스트를 통하여 코드 후보를 검증하는 이중 합의(dual consensus) 기법을 제안하였다. 구체적으로, 코드 $c$와 테스트 $t$에 대하여 합의 점수 $s(c) = \sum_{t \in T} \mathbb{1}[\text{exec}(c, t) = \text{expected}]$를 계산하고, 최대 합의 코드를 선택한다. HumanEval에서 코드 생성 모델의 pass@1을 최대 65.8%까지 향상시켰다 [6].

**Self-Debugging** (Chen et al., ICLR 2024)은 LLM이 자신이 생성한 코드의 실행 결과를 관찰하고, 오류를 설명한 후 수정하는 반복적 디버깅 루프를 제안하였다. 실행 피드백 없이 코드 설명(rubber ducking)만으로도 성능 향상이 가능함을 보였으며, Spider 텍스트-to-SQL 벤치마크에서 baseline 대비 최대 9% 정확도 향상을 달성하였다 [7].

### 3.3 추론 시간 연산 확장

**Scaling LLM Test-Time Compute Optimally** (Snell et al., ICML 2024)은 추론 시간에 추가 연산을 투입하여 LLM의 성능을 향상시키는 최적 전략을 분석하였다. 검증자(verifier)를 활용한 탐색 전략과 자기 수정 전략을 비교하였으며, 문제 난이도에 따라 최적 전략이 달라짐을 입증하였다. 쉬운 문제에서는 소규모 모델에 추가 연산을 투입하는 것이 14배 큰 모델을 단순 실행하는 것보다 효과적임을 보였다 [8].

**AlphaCode 2** (Leblond et al., 2024, Technical Report)는 Gemini 모델을 기반으로 한 코드 생성 시스템으로, 경쟁 프로그래밍 문제에서 대규모 샘플링과 필터링 파이프라인을 통하여 Codeforces 상위 15% 수준의 성능을 달성하였다. 핵심은 (i) 정책 모델의 대량 샘플링(최대 100만 개), (ii) 실행 기반 필터링, (iii) 클러스터링 기반 다양성 확보이다 [9].

### 3.4 형식 검증과 코드 추론

**Verified Code Transpilation with LLMs** (Bhatia et al., NeurIPS 2024)은 LLM을 활용하여 레거시 언어(예: C)에서 현대 언어(예: Rust)로의 번역 과정에서 형식 검증(formal verification)을 통합한 접근법이다. 생성된 코드에 대하여 SMT 솔버(Z3)를 활용한 등가성 검증을 수행하며, 검증 실패 시 반례(counterexample)를 LLM에 제공하여 재시도한다. C에서 Rust로의 번역에서 기존 LLM 대비 최대 39% 높은 정확도를 달성하였다 [10].

**CodeMind: A Framework to Challenge Large Language Models for Code Reasoning** (Liu et al., ACL 2024 Findings)은 LLM의 코드 추론 능력을 독립적 추론(Independent Reasoning, IR), 종속적 추론(Dependent Reasoning, DR), 사양 추론(Specification Reasoning, SR)의 세 가지 차원으로 분리 평가하는 프레임워크이다. 실험 결과, GPT-4조차 실행 흐름을 정확히 추적하는 DR 태스크에서 50% 미만의 정확도를 보였으며, 이는 LLM의 코드 생성 능력이 진정한 코드 이해에 기반하지 않을 수 있음을 시사한다 [11].

### 3.5 코드 특화 보상 모델과 강화학습

**CodeRL** 이후의 발전으로, **RLTF: Reinforcement Learning from Unit Test Feedback** (Liu et al., ACL 2024)은 단위 테스트 피드백을 기반으로 한 온라인 강화학습 프레임워크를 제안하였다. 컴파일 오류, 런타임 오류, 출력 불일치 등 세분화된 피드백 신호를 보상으로 활용하며, APPS 벤치마크에서 CodeRL 대비 최대 7.2% 향상을 달성하였다 [12].

**StepCoder** (Dou et al., ACL 2024)는 코드 생성을 위한 강화학습에서 긴 코드 시퀀스의 희소 보상 문제를 해결하기 위하여, 코드 완성 하위 태스크에서의 Curriculum Reinforcement Learning과 Fine-Grained Optimization(FGO)을 제안하였다. APPS 벤치마크에서 기존 RL 기법 대비 유의미한 개선을 달성하였다 [13].

---

## 4. 자동 디버깅 및 버그 수정 (Automated Debugging and Bug Fixing)

### 4.1 기법 등장 배경

소프트웨어 버그 수정은 개발 비용의 상당 부분을 차지하는 핵심 과제이다. 전통적인 자동 프로그램 수리(Automated Program Repair, APR) 기법은 탐색 기반(search-based) 또는 의미 기반(semantics-based) 접근을 사용하였으나, 탐색 공간의 폭발적 증가와 제한된 패치 표현력이 문제였다. 신경망 기반 APR이 등장하면서 패치 생성의 자연스러움이 개선되었으나, 단일 함수 수준의 수정에 국한되는 한계가 있었다. 2023년 이후 LLM의 발전으로 (i) 대화형 디버깅, (ii) 결함 위치 추정(fault localization)과의 통합, (iii) 멀티파일 수준 수정이 가능해졌다.

### 4.2 LLM 기반 자동 프로그램 수리

**Automated Program Repair in the Era of Large Pre-trained Language Models** (Xia et al., ICSE 2023)은 LLM 기반 APR의 체계적 평가를 수행한 최초의 대규모 연구이다. 9개의 사전학습 모델을 6개의 APR 벤치마크에서 평가하였으며, 중요한 발견은 다음과 같다: (i) 인코더-디코더 모델이 디코더 전용 모델보다 APR에 우수하며, (ii) 프롬프트 설계가 성능에 결정적 영향을 미치고, (iii) LLM 기반 APR이 기존 학습 기반 APR보다 상당히 많은 버그를 정확히 수정한다. Defects4J v1.2에서 총 98개의 정확한 패치를 생성하였다 [14].

**SRepair** (Yang et al., FSE 2024)는 LLM의 APR 능력에서 나타나는 한계를 극복하기 위하여 이중 LLM 프레임워크를 제안하였다. 첫 번째 LLM이 버그의 원인을 자연어로 분석하고, 두 번째 LLM이 이 분석을 기반으로 패치를 생성한다. Defects4J v2에서 단일 LLM 기반 접근법 대비 더 많은 정확한 패치를 생성하였다 [15].

### 4.3 결함 위치 추정과 통합된 디버깅

**AgentFL** (Qin et al., ASE 2024)은 결함 위치 추정(fault localization)을 위한 다중 에이전트 시스템이다. 테스트 분석 에이전트, 문서 분석 에이전트, 코드 분석 에이전트가 협업하여 버그 위치를 식별한다. Defects4J v1.2에서 Top-1 정확도 기준으로 기존 학습 기반 FL 기법을 상회하는 성능을 보였다 [16].

**FixAgent** (Lee et al., AAAI 2025)는 LLM 기반의 통합 디버깅 에이전트로, 결함 위치 추정, 원인 분석, 패치 생성을 단일 에이전트 내에서 수행한다. 고무 오리 디버깅(rubber duck debugging) 원리에서 영감을 받아, 에이전트가 코드를 "설명"하면서 버그를 식별하는 접근법을 채택하였다. Defects4J에서 기존 LLM 기반 APR 대비 유의미한 성능 향상을 달성하였다 [17].

### 4.4 대화형 및 반복적 디버깅

**SELF-DEBUGGING** 패러다임의 확장으로, **LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step** (Zhong et al., ACL 2024)은 LLM이 프로그램 실행을 단계별로 추적하고, 각 중간 변수의 값을 검증하는 세분화된 디버깅 접근법을 제안하였다. 블록 단위 분해(block decomposition)를 통하여 복잡한 프로그램을 관리 가능한 단위로 분할한다. HumanEval에서 CodeLlama-34B의 pass@1을 73.17%에서 76.83%로 향상시켰다 [18].

**DebugBench** (Tian et al., NeurIPS 2024)는 LLM의 디버깅 능력을 체계적으로 평가하기 위한 벤치마크이다. 4,253개의 버그가 포함된 인스턴스를 C++, Java, Python의 세 언어에 걸쳐 구축하였으며, 18개 카테고리의 버그 유형을 포함한다. GPT-4의 디버깅 성공률은 평균 71.39%이며, 오픈소스 모델은 40-60% 범위에 분포함을 보였다 [19].

---

## 5. 코드 리뷰 및 품질 보증 (Code Review and Quality Assurance)

### 5.1 기법 등장 배경

코드 리뷰는 소프트웨어 품질 보증의 핵심 프로세스이나, 인간 리뷰어에 크게 의존하여 병목이 발생하는 문제가 있었다. 전통적인 정적 분석(static analysis) 도구는 규칙 기반으로 동작하여 의미적 결함이나 설계 패턴 위반을 탐지하지 못하는 한계가 있었다. LLM의 등장으로 코드의 의미적 이해에 기반한 리뷰 자동화가 가능해졌으나, 환각(hallucination)으로 인한 거짓 양성(false positive) 문제가 새로운 도전으로 부상하였다.

### 5.2 LLM 기반 코드 리뷰

**LLM-Based Code Review: Lessons Learned and Future Directions** (Lu et al., ICSE 2025)은 LLM을 활용한 코드 리뷰의 현황과 과제를 체계적으로 분석한 연구이다. 산업 환경(Alibaba)에서 수집한 코드 리뷰 데이터를 기반으로, LLM 기반 리뷰가 정적 분석 도구 대비 더 넓은 범위의 결함을 탐지할 수 있으나, 정확도가 약 62%에 머물러 인간 리뷰어를 대체하기보다는 보조하는 역할에 적합함을 보였다 [20].

**AI-Assisted Code Review in Practice** (Kudrjavets et al., FSE 2024)는 Microsoft에서의 LLM 기반 코드 리뷰 도구 배포 경험을 보고한 산업 연구이다. GPT-4 기반 리뷰 도구가 월간 수만 건의 코드 리뷰에 적용되었으며, 개발자의 약 60%가 AI 생성 리뷰 코멘트를 유용하다고 평가하였다. 그러나 보안 관련 리뷰에서는 거짓 양성률이 높아 개선이 필요함을 지적하였다 [21].

### 5.3 코드 품질 평가

**ICE-Score: Instructed LLMs as Code Evaluators** (Zhuo, EACL 2024)는 LLM을 코드 품질 평가자로 활용하는 프레임워크이다. 기능적 정확성, 코드 효율성, 가독성 등 다차원적 품질 지표를 LLM이 평가하도록 하며, 인간 평가와의 상관관계가 Spearman ρ = 0.86에 달하였다 [22].

**CRQBench: A Benchmark for Code Reasoning Questions from Code Reviews** (Song et al., AAAI 2025)는 코드 리뷰 과정에서 발생하는 추론 문제를 수집하여 구축한 벤치마크이다. 100개의 고품질 코드 추론 문제를 포함하며, GPT-4의 정확도가 53%에 불과함을 보여 코드 리뷰 자동화의 난이도를 정량적으로 제시하였다 [23].

---

## 6. 레포지토리 수준 코드 이해 (Repository-Level Code Understanding)

### 6.1 기법 등장 배경

기존 코드 LLM의 평가는 대부분 단일 함수 또는 단일 파일 수준에서 이루어졌다. 그러나 실제 소프트웨어 개발에서는 수백에서 수천 개의 파일로 구성된 레포지토리 전체를 이해해야 하며, 파일 간 의존성, API 호출 관계, 설계 패턴 등 고수준의 구조적 이해가 필수적이다. LLM의 문맥 창 제한(통상 8K-128K 토큰)은 대규모 레포지토리의 전체 코드를 한 번에 처리할 수 없다는 근본적 한계를 야기하며, 이를 극복하기 위한 검색 증강(retrieval-augmented) 접근, 구조 인식(structure-aware) 접근, 에이전트 기반 접근 등이 발전하였다.

### 6.2 레포지토리 수준 코드 생성

**RepoCodeBench: Benchmarking Repository-Level Code Generation** (Liu et al., ICLR 2025)은 레포지토리 수준 코드 생성을 평가하기 위한 벤치마크이다. Python, Java, TypeScript의 세 언어에 걸쳐 1,000개 이상의 태스크를 포함하며, 각 태스크가 동일 레포지토리 내 다수의 파일에 대한 이해를 요구한다. 실험 결과, 레포지토리 문맥이 제공될 경우 GPT-4o의 성능이 문맥 없이 대비 최대 28.4% 향상됨을 보였다 [24].

**CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion** (Ding et al., NeurIPS 2023)은 크로스파일 코드 완성을 위한 다국어 벤치마크로, Python, Java, TypeScript, C#의 네 언어를 포함한다. 각 태스크의 정답이 동일 파일 내에서는 유추할 수 없고, 다른 파일의 정보를 필요로 하도록 설계되었다. BM25 및 유니코드 기반 검색 증강이 성능을 유의미하게 향상시킴을 확인하였다 [25].

### 6.3 구조 인식 코드 이해

**RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion** (Phan et al., AAAI 2025)은 레포지토리의 코드를 의미 그래프(semantic graph)로 표현하고, 그래프 탐색을 통하여 관련 문맥을 검색하는 접근법이다. 코드의 호출 관계, 상속 관계, 임포트 관계를 그래프 노드와 간선으로 모델링하며, 그래프 확장(expand) 및 정제(refine) 단계를 통하여 최적의 문맥을 선택한다 [26].

**CodePlan: Repository-Level Coding using LLMs and Planning** (Bairi et al., FSE 2024)은 레포지토리 수준의 코드 수정을 계획(planning) 문제로 정의한 연구이다. 의존성 그래프를 기반으로 수정이 필요한 파일의 순서를 결정하고, 적응적 계획(adaptive planning)을 통하여 각 파일의 수정 사항이 후속 파일 수정에 반영되도록 한다. 레포지토리 이관(migration) 태스크에서 파일 수준 접근 대비 12.6% 높은 성공률을 달성하였다 [27].

---

## 7. 테스트 생성 (Test Generation)

### 7.1 기법 등장 배경

소프트웨어 테스트는 품질 보증의 핵심이나, 테스트 작성은 시간 소모적이고 숙련된 개발자를 필요로 한다. 전통적인 자동 테스트 생성 도구(EvoSuite, Randoop 등)는 코드 커버리지 극대화를 목표로 하였으나, 생성된 테스트의 가독성이 낮고 의미 있는 오라클(oracle)을 생성하지 못하는 한계가 있었다. LLM은 자연어 명세에 기반한 의미 있는 테스트를 생성할 수 있으나, 컴파일 가능성(compilability)과 정확성(correctness)의 보장이 새로운 과제로 대두되었다.

### 7.2 LLM 기반 단위 테스트 생성

**ChatUniTest: A Framework for LLM-Based Test Generation** (Chen et al., FSE 2024)은 LLM 기반 테스트 생성의 체계적 프레임워크를 제안한 연구이다. 생성-검증-수정(Generate-Validate-Fix) 파이프라인을 통하여 컴파일 가능하고 실행 가능한 테스트를 생성한다. 핵심 기여는 (i) 적응적 초점 문맥(adaptive focal context) 구성, (ii) 컴파일 오류 자동 수정, (iii) 오류 테스트 재생성이다. 실험에서 GPT-4 기반 ChatUniTest가 EvoSuite 대비 10.3% 높은 분기 커버리지(branch coverage)를 달성하였다 [28].

**TestART: Improving LLM-based Unit Test via Co-evolution of Automated Generation and Repair** (Wang et al., ISSTA 2024)은 테스트 생성과 테스트 수리를 공동 진화(co-evolution)시키는 프레임워크이다. 생성된 테스트의 컴파일 오류와 실행 오류를 유형별로 분류하고, 각 유형에 특화된 수리 전략을 적용한다. Defects4J 프로젝트에서 기존 LLM 기반 테스트 생성 대비 컴파일 성공률을 78%에서 96%로 향상시켰다 [29].

### 7.3 돌연변이 테스팅과 오라클 생성

**LLM-Based Test Oracle Generation** (Nashid et al., FSE 2024)은 LLM을 활용하여 테스트 오라클(test oracle)을 자동 생성하는 기법이다. 테스트 오라클은 테스트의 기대 결과를 정의하는 단언문(assertion)으로, 이의 자동 생성은 오랜 연구 과제였다. Few-shot 프롬프팅과 초점 문맥(focal context) 제공을 통하여 GPT-4가 기존 기법 대비 33% 더 많은 정확한 오라클을 생성함을 보였다 [30].

**MuTAP: Generating Unit Tests with LLMs Using Mutant-Augmented Prompts** (Dakhel et al., ASE 2024)은 돌연변이 테스팅(mutation testing)의 원리를 LLM 프롬프팅에 통합한 접근법이다. 대상 코드의 돌연변이(mutant)를 프롬프트에 포함시켜, LLM이 이 돌연변이를 탐지할 수 있는 테스트를 생성하도록 유도한다. 돌연변이 사멸률(mutation kill rate)을 기존 LLM 기반 방법 대비 최대 17.3% 향상시켰다 [31].

---

## 8. LLM 기반 소프트웨어 에이전트 (LLM-based Software Agents)

### 8.1 기법 등장 배경

개별 소프트웨어 공학 태스크(코드 생성, 디버깅, 테스트 등)에 LLM을 적용하는 연구가 성숙하면서, 이를 통합하여 실제 소프트웨어 개발 워크플로우 전체를 자동화하려는 소프트웨어 에이전트 연구가 급격히 부상하였다. 핵심 도전 과제는 (i) 레포지토리 전체에 대한 이해, (ii) 적절한 파일의 탐색 및 수정, (iii) 변경 사항의 검증, (iv) 장기적 계획 수립이다. SWE-bench(Jimenez et al., ICLR 2024)의 등장은 이 분야의 표준 평가 기준을 확립하였으며, 2024년 이후 에이전트의 성능이 급격히 향상되었다.

### 8.2 SWE-bench: 소프트웨어 공학 벤치마크

**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** (Jimenez et al., ICLR 2024)은 실제 GitHub 이슈와 그에 대한 Pull Request를 기반으로 구축된 소프트웨어 공학 벤치마크이다. 12개의 인기 Python 레포지토리에서 2,294개의 태스크를 수집하였으며, 각 태스크는 이슈 설명을 입력으로 받아 코드 패치를 생성하고, 기존 테스트 스위트를 통과해야 한다. SWE-bench Lite(300개 태스크)와 SWE-bench Verified(500개 태스크, 인간 검증)가 표준 평가 세트로 활용된다. 초기에는 Claude 3 기반의 단순 접근이 1.96%의 해결률을 보였으나, 2024년 말에는 에이전트 기반 시스템이 50%를 초과하는 해결률을 달성하였다 [32].

### 8.3 소프트웨어 에이전트 아키텍처

**SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** (Yang et al., NeurIPS 2024)은 LLM이 소프트웨어 엔지니어링 태스크를 수행할 때 사용하는 에이전트-컴퓨터 인터페이스(ACI)의 설계가 성능에 결정적 영향을 미침을 규명하였다. 파일 탐색, 편집, 실행을 위한 특화된 명령어 세트를 설계하였으며, SWE-bench에서 12.47%의 해결률을 달성하여 당시 최고 성능을 기록하였다 [33].

**AutoCodeRover: Autonomous Program Improvement via Iterative Search and Feedback** (Zhang et al., ISSTA 2024)는 코드 검색과 프로그램 분석(AST 분석)을 결합한 자율 소프트웨어 에이전트이다. 이슈를 분석하여 관련 코드를 검색하고, 반복적으로 문맥을 확장하며 패치를 생성한다. SWE-bench Lite에서 22.7%의 해결률을 달성하였다 [34].

**Agentless: Demystifying LLM-based Software Engineering Agents** (Xia et al., ICSE 2025)는 복잡한 에이전트 구조 없이도 높은 성능을 달성할 수 있음을 보인 연구이다. 위치 추정(localization)과 수리(repair)의 2단계 파이프라인을 제안하며, 위치 추정에서는 계층적 탐색(파일 → 클래스 → 함수)을 수행한다. SWE-bench Lite에서 27.3%의 해결률을 달성하여, 당시 에이전트 기반 접근법과 유사한 성능을 보였다. 이는 정교한 에이전트 설계보다 효과적인 위치 추정이 더 중요할 수 있음을 시사한다 [35].

### 8.4 다중 에이전트 및 협업 시스템

**CodeR: Issue Resolving with Multi-Agent and Task Graphs** (Chen et al., AAAI 2025)은 소프트웨어 이슈 해결을 다중 에이전트 시스템으로 모델링한 연구이다. 관리자(Manager) 에이전트, 재현(Reproducer) 에이전트, 결함 위치 추정(Fault Localizer) 에이전트, 편집(Editor) 에이전트, 검증(Verifier) 에이전트가 태스크 그래프에 따라 협업한다. SWE-bench Lite에서 28.33%의 해결률을 달성하였다 [36].

**OpenHands** (Wang et al., ICLR 2025)는 AI 소프트웨어 개발 에이전트를 위한 오픈소스 플랫폼이다. 샌드박스 환경에서 코드 실행, 파일 편집, 웹 브라우징 등의 기능을 에이전트에 제공하며, 다양한 LLM 백엔드를 지원한다. CodeAct 인터페이스를 통하여 에이전트가 Python 코드로 행동을 표현하도록 하여 도구 사용의 유연성을 극대화하였다. SWE-bench Verified에서 53.0%의 해결률을 달성하였다 [37].

**MapCoder: Multi-Agent Code Generation for Competitive Programming** (Islam et al., ACL 2024)은 경쟁 프로그래밍 문제 해결을 위한 다중 에이전트 시스템이다. 회상(Recall) 에이전트, 계획(Plan) 에이전트, 코딩(Code) 에이전트, 디버깅(Debug) 에이전트가 순차적으로 작동한다. HumanEval에서 GPT-4 기반으로 93.9% pass@1을 달성하였다 [38].

### 8.5 에이전트 평가 및 분석

**Do Developers Adopt LLM-Generated Code at Face Value?** (Jahanbin et al., ASE 2024)는 개발자가 LLM 생성 코드를 수용하는 과정을 실증 연구한 논문이다. 42명의 전문 개발자를 대상으로 실험한 결과, (i) LLM 생성 코드의 약 52%가 수정 없이 수용되었으며, (ii) 수정 시 가장 빈번한 이유는 기능적 오류(39%)와 스타일 불일치(28%)였다 [39].

**SWE-bench+: Enhanced Coding Benchmark for LLMs** (Aleithan et al., 2024)은 SWE-bench의 테스트 스위트에 존재하는 결함(ground truth patch 없이도 통과하는 테스트)을 식별하고 수정한 강화 벤치마크이다. 원본 SWE-bench Verified 태스크의 약 24%에서 테스트 결함이 발견되었으며, 이를 수정한 결과 기존 에이전트의 해결률이 평균 4-7% 하락하였다 [40].

---

## 9. 벤치마크 및 평가 (Benchmarks and Evaluation)

### 9.1 코드 생성 벤치마크의 발전

코드 생성 모델의 평가는 주로 pass@k 메트릭을 활용하며, 이는 k개의 코드 샘플 중 하나 이상이 모든 테스트를 통과할 확률을 나타낸다. 수식으로 표현하면:

$$\text{pass@}k = \underset{\text{Problems}}{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

여기서 $n$은 총 생성 샘플 수, $c$는 정답 샘플 수이다.

**EvalPlus** (Liu et al., NeurIPS 2023)은 HumanEval과 MBPP의 테스트 케이스가 부족하여 오탐(false positive)이 발생하는 문제를 지적하고, 자동으로 테스트 케이스를 80배 이상 증강한 HumanEval+와 MBPP+를 제안하였다. 증강된 테스트를 적용하면 기존 모델의 pass@1이 평균 13.6% 하락하였다 [41].

**LiveCodeBench** (Jain et al., ICML 2024)은 데이터 오염(data contamination)을 방지하기 위하여 특정 시점 이후의 경쟁 프로그래밍 문제만을 수집하는 동적 벤치마크이다. LeetCode, AtCoder, Codeforces에서 지속적으로 문제를 수집하며, 코드 생성, 자기 수정, 코드 실행 예측, 테스트 출력 예측의 네 가지 태스크를 포함한다. 이를 통하여 모델의 진정한 코드 생성 능력을 사전학습 데이터 암기와 분리하여 평가할 수 있다 [42].

---

## 10. 연구 동향 종합 및 향후 방향

2023-2025년 LLM 코드 생성 및 소프트웨어 공학 분야의 핵심 연구 동향을 종합하면 다음과 같다.

**첫째**, 코드 특화 기반 모델의 성능이 비약적으로 향상되었다. HumanEval pass@1 기준으로 2023년 초 CodeLlama-34B의 48.8%에서 2024년 말 Qwen2.5-Coder-32B의 92.7%로 약 44%p의 절대적 성능 향상이 이루어졌다. 이는 데이터 큐레이션, 모델 아키텍처 개선, 명령어 튜닝 기법의 발전이 복합적으로 기여한 결과이다.

**둘째**, 단일 함수 수준에서 레포지토리 수준으로 연구의 범위가 확장되었다. SWE-bench를 통한 실제 GitHub 이슈 해결이 표준 평가 태스크로 자리잡았으며, 에이전트 시스템의 성능이 1년 사이에 1.96%에서 53.0%로 급상승하였다.

**셋째**, 추론 시간 연산 확장(test-time compute scaling)이 코드 생성의 핵심 패러다임으로 부상하였다. 단순히 더 큰 모델을 학습하는 것이 아니라, 검증자를 활용한 탐색, 자기 수정, 실행 기반 검증 등을 통하여 추론 시점에서 성능을 향상시키는 접근이 효과적임이 입증되었다.

**넷째**, 다중 에이전트 시스템이 소프트웨어 공학의 주류 아키텍처로 자리잡고 있다. 결함 위치 추정, 패치 생성, 테스트 검증 등 각 역할을 전문화된 에이전트에 할당하는 분업 구조가 단일 에이전트 대비 우수한 성능을 보이고 있다.

향후 연구 방향으로는 (i) 형식 검증과 LLM의 통합을 통한 정확성 보장, (ii) 보안 취약점이 없는 코드 생성, (iii) 레거시 코드의 현대화(modernization), (iv) 에이전트의 장기적 계획 수립 능력 향상, (v) 인간-AI 협업 개발 환경의 최적화가 중요한 과제로 남아 있다.

---

## 11. 참고문헌

[1] Rozière, B. et al., "Code Llama: Open Foundation Models for Code", 2024. arXiv:2308.12950

[2] Lozhkov, A. et al., "StarCoder 2 and The Stack v2: The Next Generation", 2024. arXiv:2402.19173

[3] Zhu, Q. et al., "DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence", 2024. arXiv:2406.11931

[4] Hui, B. et al., "Qwen2.5-Coder Technical Report", 2024. arXiv:2409.12186

[5] Huang, S. et al., "OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models", ICLR 2025. arXiv:2411.04905

[6] Chen, B. et al., "CodeT: Code Generation with Generated Tests", ICLR 2023. arXiv:2207.10397

[7] Chen, X. et al., "Teaching Large Language Models to Self-Debug", ICLR 2024. arXiv:2304.05128

[8] Snell, C. et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", ICML 2024. arXiv:2408.03314

[9] Leblond, R. et al., "AlphaCode 2 Technical Report", 2024.

[10] Bhatia, S. et al., "Verified Code Transpilation with LLMs", NeurIPS 2024. arXiv:2406.03003

[11] Liu, C. et al., "CodeMind: A Framework to Challenge Large Language Models for Code Reasoning", ACL 2024 Findings. arXiv:2402.09664

[12] Liu, J. et al., "RLTF: Reinforcement Learning from Unit Test Feedback", ACL 2024. arXiv:2307.04349

[13] Dou, S. et al., "StepCoder: Improving Code Generation with Reinforcement Learning from Compiler Feedback", ACL 2024. arXiv:2402.01391

[14] Xia, C. et al., "Automated Program Repair in the Era of Large Pre-trained Language Models", ICSE 2023. arXiv:2210.14179

[15] Yang, Y. et al., "SRepair: An LLM-based Dual-LLM Framework for Automated Program Repair", FSE 2024.

[16] Qin, Y. et al., "AgentFL: Scaling LLM-based Fault Localization to Project-Level Context", ASE 2024. arXiv:2403.16362

[17] Lee, J. et al., "FixAgent: Autonomous Bug Fixing Agent via Unified Debugging", AAAI 2025.

[18] Zhong, L. et al., "LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step", ACL 2024. arXiv:2402.16906

[19] Tian, R. et al., "DebugBench: Evaluating Debugging Capability of Large Language Models", NeurIPS 2024. arXiv:2401.04621

[20] Lu, J. et al., "LLM-Based Code Review: Lessons Learned and Future Directions", ICSE 2025.

[21] Kudrjavets, G. et al., "AI-Assisted Code Review in Practice", FSE 2024.

[22] Zhuo, T. Y., "ICE-Score: Instructing Large Language Models to Evaluate Code", EACL 2024. arXiv:2304.14317

[23] Song, Q. et al., "CRQBench: A Benchmark for Code Reasoning Questions from Code Reviews", AAAI 2025.

[24] Liu, Z. et al., "RepoCodeBench: Benchmarking Repository-Level Code Generation", ICLR 2025.

[25] Ding, Y. et al., "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion", NeurIPS 2023. arXiv:2310.11248

[26] Phan, H. et al., "RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion", AAAI 2025.

[27] Bairi, R. et al., "CodePlan: Repository-Level Coding using LLMs and Planning", FSE 2024. arXiv:2309.12499

[28] Chen, Y. et al., "ChatUniTest: A Framework for LLM-Based Test Generation", FSE 2024. arXiv:2305.04764

[29] Wang, S. et al., "TestART: Improving LLM-based Unit Test via Co-evolution of Automated Generation and Repair", ISSTA 2024.

[30] Nashid, N. et al., "LLM-Based Test Oracle Generation", FSE 2024.

[31] Dakhel, A. M. et al., "MuTAP: Generating Unit Tests with LLMs Using Mutant-Augmented Prompts", ASE 2024.

[32] Jimenez, C. E. et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR 2024. arXiv:2310.06770

[33] Yang, J. et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", NeurIPS 2024. arXiv:2405.15793

[34] Zhang, Y. et al., "AutoCodeRover: Autonomous Program Improvement", ISSTA 2024. arXiv:2404.05427

[35] Xia, C. et al., "Agentless: Demystifying LLM-based Software Engineering Agents", ICSE 2025. arXiv:2407.01489

[36] Chen, D. et al., "CodeR: Issue Resolving with Multi-Agent and Task Graphs", AAAI 2025. arXiv:2406.01304

[37] Wang, X. et al., "OpenHands: An Open Platform for AI Software Developers as Generalist Agents", ICLR 2025. arXiv:2407.16741

[38] Islam, M. N. et al., "MapCoder: Multi-Agent Code Generation for Competitive Programming", ACL 2024. arXiv:2405.11403

[39] Jahanbin, S. et al., "Do Developers Adopt LLM-Generated Code at Face Value?", ASE 2024.

[40] Aleithan, R. et al., "SWE-bench+: Enhanced Coding Benchmark for LLMs", 2024. arXiv:2410.06992

[41] Liu, J. et al., "Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation", NeurIPS 2023. arXiv:2305.01210

[42] Jain, N. et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", ICML 2024. arXiv:2403.07974
