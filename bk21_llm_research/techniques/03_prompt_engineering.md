# Prompt Engineering (프롬프트 엔지니어링)

## 1. 기법의 정의

Prompt Engineering은 대규모 언어 모델(LLM)의 파라미터를 수정하지 않고, 입력 텍스트(프롬프트)의 구조, 내용, 형식을 체계적으로 설계하여 모델의 출력 품질을 최적화하는 기법이다. 형식적으로, 언어 모델 $M$이 입력 $x$에 대해 출력 $y$를 생성하는 함수 $M: X \rightarrow Y$일 때, Prompt Engineering은 프롬프트 함수 $p: X \rightarrow X'$를 설계하여 $M(p(x))$의 성능을 최대화하는 최적화 문제로 정의된다:

$$p^* = \arg\max_{p \in \mathcal{P}} \mathbb{E}_{(x,y) \sim \mathcal{D}} [\text{metric}(M(p(x)), y)]$$

여기서 $\mathcal{P}$는 가능한 프롬프트 변환의 공간이며, $\mathcal{D}$는 태스크 데이터 분포이다. 이 정의는 Fine-tuning이 $M$의 파라미터 $\theta$를 최적화하는 것과 대조적으로, Prompt Engineering은 $p$만을 최적화한다는 점에서 근본적으로 상이하다.

---

## 2. 기존 기법의 한계와 Prompt Engineering 등장 배경

### 2.1 Supervised Fine-tuning의 한계

전통적 NLP 파이프라인에서 태스크 적응은 Supervised Fine-tuning(SFT)에 의존하였다. SFT는 사전 학습된 모델의 전체 파라미터를 태스크별 레이블 데이터로 재학습하는 방식이다. 그러나 다음과 같은 구조적 한계가 존재하였다:

1. **데이터 요구량**: 태스크당 수천에서 수만 개의 레이블 데이터가 필요하며, 도메인 특화 데이터의 확보 비용이 높다.
2. **계산 비용**: GPT-3 (175B) 규모의 모델을 Fine-tuning하는 데 수백 GPU-hours가 소요되며, 이는 대부분의 연구 그룹과 기업에게 비현실적이다.
3. **일반화 저하**: Fine-tuning은 catastrophic forgetting 현상을 유발하여, 특정 태스크에 특화되면 다른 태스크의 성능이 하락한다 (Kirkpatrick et al., **PNAS**, 2017).
4. **API 접근 환경의 확산**: OpenAI, Anthropic 등의 API 서비스가 확산되면서, 모델 파라미터에 직접 접근할 수 없는 black-box 환경이 보편화되었다.

### 2.2 In-Context Learning의 발견

Brown et al. (2020)은 GPT-3 논문에서 대규모 언어 모델이 프롬프트에 포함된 소수의 예시만으로 새로운 태스크를 수행할 수 있음을 실증하였다. 이는 In-Context Learning(ICL)이라 명명되었으며, 모델의 파라미터 업데이트 없이 추론 시점에서 태스크 적응이 가능하다는 패러다임 전환을 의미하였다. GPT-3는 few-shot 설정에서 SuperGLUE 벤치마크의 다수 태스크에서 Fine-tuned BERT 모델과 경쟁 가능한 성능을 달성하였다.

### 2.3 프롬프트 민감성 문제

그러나 ICL의 발견은 동시에 심각한 취약성을 드러내었다. Lu et al. (2022)은 few-shot 예시의 순서만 변경해도 SST-2에서 정확도가 54.3%에서 93.4%까지 변동함을 보고하였다. 이는 프롬프트 설계가 단순한 시행착오가 아닌, 체계적 방법론을 요구하는 연구 영역임을 시사하였다. 이러한 배경에서 Prompt Engineering이 독립적 연구 분야로 자리잡게 되었다.

---

## 3. 주요 기법 상세

### 3.1 Zero-shot Prompting

**정의**: 태스크에 대한 예시를 제공하지 않고, 태스크 설명(instruction)만으로 모델의 수행을 유도하는 기법이다.

**이전 문제**: 사전 학습된 모델은 태스크별 데이터 없이는 특정 형식의 출력을 생성하기 어렵다는 것이 통념이었다.

**해결 방식**: GPT-3 규모(175B)의 모델이 instruction-following 능력을 emergent하게 획득함이 확인되었다. Wei et al. (2022a)는 이러한 능력이 모델 규모가 임계점을 넘을 때 갑작스럽게 출현하는 emergent ability임을 보고하였다.

**실험 결과**: Kojima et al. (2022)은 "Let's think step by step"이라는 단일 문장 추가(Zero-shot CoT)만으로 MultiArith에서 정확도를 17.7%에서 78.7%로, GSM8K에서 10.4%에서 40.7%로 향상시킴을 보고하였다(InstructGPT 기준).

**잔존 한계**: Zero-shot은 복잡한 추론이나 도메인 특화 태스크에서 여전히 few-shot 대비 성능이 낮으며, 모델이 태스크의 의도를 오해할 확률이 높다.

### 3.2 Few-shot Prompting (In-Context Learning)

**정의**: 프롬프트 내에 소수의 입출력 예시(demonstrations)를 포함하여 모델이 태스크 패턴을 추론 시점에서 학습하도록 유도하는 기법이다.

**이전 문제**: Zero-shot prompting은 태스크의 형식을 명시적으로 전달하지 못하여, 모델이 원하는 출력 형식을 생성하지 못하는 경우가 빈번하였다.

**해결 방식**: Brown et al. (2020)은 GPT-3에서 프롬프트에 $k$개($k$ = 1~64)의 예시를 포함하면, 모델이 해당 패턴을 in-context에서 학습하여 태스크를 수행할 수 있음을 실증하였다. 형식적으로:

$$\text{prompt} = [d_1, d_2, \ldots, d_k, x_{\text{query}}], \quad d_i = (x_i, y_i)$$

**실험 결과**: GPT-3 175B는 few-shot(32-shot) 설정에서 TriviaQA에서 71.2%, LAMBADA에서 86.4%의 정확도를 달성하였으며, 이는 Fine-tuned 모델과 경쟁 가능한 수준이다.

**잔존 한계**:
- **예시 선택 민감성**: Liu et al. (2022)은 예시 선택 방법에 따라 성능이 최대 30% 변동함을 보고하였다.
- **예시 순서 민감성**: Lu et al. (2022)은 순서에 따른 분산이 극도로 높음을 실증하였다.
- **토큰 비용**: 예시 포함으로 인한 입력 토큰 증가는 비용과 지연 시간을 증가시킨다.
- **추론 능력의 부재**: 수학적 추론, 다단계 논리 추론에서는 예시만으로 성능 향상이 제한적이다.

### 3.3 Chain-of-Thought (CoT) Prompting

> 본 섹션은 CoT의 핵심 개념을 Prompt Engineering의 맥락에서 요약한다. CoT의 심층 분석, 변형 기법, 및 Reasoning 모델(o1, DeepSeek-R1)과의 관계는 `11_chain_of_thought.md`를 참조하라.

**이전 문제**: Few-shot prompting은 패턴 매칭 수준의 태스크에서는 효과적이나, 다단계 수학적 추론(예: GSM8K)이나 논리 추론(예: StrategyQA)에서는 직접 답변(direct answer) 생성 방식의 한계로 인해 성능이 급격히 하락하였다.

**해결 방식**: Wei et al. (2022b)은 프롬프트의 예시에 중간 추론 단계(intermediate reasoning steps)를 포함하면, 모델이 유사한 추론 체인을 생성하여 복잡한 문제를 분해할 수 있음을 발견하였다.

$$\text{prompt} = [(x_1, r_1, y_1), \ldots, (x_k, r_k, y_k), x_{\text{query}}]$$

여기서 $r_i$는 $x_i$에서 $y_i$에 이르는 중간 추론 과정이다.

**실험 결과**: PaLM 540B에서 CoT 적용 시 GSM8K 정확도가 17.9%에서 58.1%로 향상되었다. 이는 표준 few-shot prompting 대비 40.2%p의 개선이다. 중요한 관찰은 이 효과가 모델 규모에 의존한다는 점이다. 약 100B 파라미터 이하의 모델에서는 CoT가 오히려 성능을 하락시키는 경우가 관찰되었다.

**잔존 한계**: 단일 추론 경로에 의존하여, 초기 단계의 오류가 후속 단계로 전파(error propagation)되는 문제가 존재한다.

### 3.4 Self-Consistency

**이전 문제**: CoT는 단일 greedy decoding 경로를 생성하므로, 해당 경로에 오류가 포함되면 최종 답변 역시 오류를 포함한다. 이는 복잡한 문제일수록 심각한 제약이다.

**해결 방식**: Wang et al. (2023a)은 Self-Consistency 기법을 제안하였다. 이는 동일 프롬프트에 대해 높은 temperature($T > 0$)로 $n$개의 독립적 CoT 경로를 샘플링한 후, 최종 답변에 대해 다수결 투표(majority voting)를 수행하는 방식이다.

$$\hat{y} = \arg\max_{y} \sum_{i=1}^{n} \mathbb{1}[y_i = y]$$

**실험 결과**: GSM8K에서 CoT 단독 대비 Self-Consistency(40 paths)는 PaLM 540B 기준 58.1%에서 74.4%로 +16.3%p의 정확도 향상을 달성하였다. Codex 기준으로는 65.6%에서 78.0%로 +12.4%p 향상이 보고되었다 (Wang et al., 2023a).

**잔존 한계**: $n$회 샘플링으로 인해 추론 비용이 $n$배 증가한다. 또한 다수결 투표는 모든 경로가 동일한 오류 패턴을 공유할 경우(systematic error) 효과가 제한적이다.

### 3.5 Tree-of-Thoughts (ToT)

**이전 문제**: CoT와 Self-Consistency는 모두 선형적(sequential) 추론 경로를 생성한다. 이로 인해 탐색(exploration)과 역추적(backtracking)이 필요한 계획(planning) 문제에서 구조적 한계가 존재하였다.

**해결 방식**: Yao et al. (2023a)은 추론 과정을 트리 구조로 확장하는 Tree-of-Thoughts를 제안하였다. 각 노드는 부분적 추론 상태(thought)이며, BFS 또는 DFS를 통해 탐색하고, 각 상태의 유망성을 LLM 자체가 평가(state evaluation)하여 가지치기(pruning)를 수행한다.

형식적으로, 상태 공간 $\mathcal{S}$에서 사고 생성기 $G(s) \rightarrow \{s'\}$, 상태 평가기 $V(s) \rightarrow \mathbb{R}$, 탐색 전략 $\sigma \in \{\text{BFS}, \text{DFS}\}$를 결합한 프레임워크이다.

**실험 결과**: Game of 24 과제에서 CoT의 성공률이 4.0%에 불과한 반면, ToT는 74.0%의 성공률을 달성하였다. 이는 CoT 대비 18.5배의 개선이다. Creative Writing 과제에서도 인간 평가 기준 ToT가 CoT 대비 유의미한 선호도 우위를 보고하였다.

**잔존 한계**: 탐색 과정에서 다수의 LLM 호출이 필요하여 계산 비용이 높다. 간단한 질의응답 태스크에서는 오히려 CoT 대비 성능 이점이 없으며 비용만 증가한다.

### 3.6 Graph-of-Thoughts (GoT)

**이전 문제**: ToT는 트리 구조로 제한되어, 서로 다른 추론 분기 간의 정보 병합(aggregation)이 불가능하다. 현실의 문제 해결 과정은 여러 부분 해결책을 결합하는 비순환 그래프(DAG) 형태에 가까운 경우가 많다.

**해결 방식**: Besta et al. (2024)은 추론 과정을 방향성 비순환 그래프(DAG)로 일반화하는 Graph-of-Thoughts를 제안하였다. GoT는 사고(thought)를 노드로, 사고 간 의존 관계를 엣지로 표현하며, 세 가지 핵심 연산을 정의한다: (1) Generation: 새로운 사고 생성, (2) Aggregation: 여러 사고를 하나로 병합, (3) Refinement: 기존 사고를 개선.

**실험 결과**: Sorting 과제에서 GoT는 ToT 대비 정렬 품질을 약 62% 향상시키면서, 비용은 약 31% 이상 절감하였다 (Besta et al., 2024).

**잔존 한계**: 그래프 구조의 설계가 태스크에 의존적이며, 범용적인 그래프 구성 전략이 부재하다. 또한 구현 복잡도가 ToT보다 높다.

### 3.7 ReAct (Reasoning + Acting)

**이전 문제**: CoT 계열 기법은 모델 내부의 추론에 국한되며, 외부 환경과의 상호작용(예: 정보 검색, 계산기 사용, API 호출)이 불가능하다. 이로 인해 최신 정보가 필요한 질의나 사실 확인이 필요한 태스크에서 hallucination이 발생한다.

**해결 방식**: Yao et al. (2023b)은 추론(Reasoning)과 행동(Acting)을 교차적으로 수행하는 ReAct 프레임워크를 제안하였다. 모델은 Thought-Action-Observation의 반복 루프를 수행한다:

```
Thought[1]: 현재 질문에 답하기 위해 X를 확인해야 한다.
Action[1]: Search["X에 대한 정보"]
Observation[1]: [검색 결과]
Thought[2]: 검색 결과에 따르면 Y이다. 따라서...
Action[2]: Finish["최종 답변"]
```

**실험 결과**: HotpotQA에서 ReAct는 standard prompting 대비 +6%의 정확도 향상을 달성하였으며, CoT 단독 대비에서도 사실적 정확성(factual accuracy)이 유의미하게 개선되었다. 특히 hallucination 비율이 CoT의 14%에서 ReAct의 6%로 감소하였다 (Yao et al., 2023b).

**잔존 한계**: Action 공간의 정의가 사전에 필요하며, 도구 호출의 오류가 전체 추론 체인을 실패시킬 수 있다. 또한 모델이 반복적 행동 루프(action loop)에 빠지는 현상이 관찰된다.

### 3.8 Reflexion

**이전 문제**: ReAct를 포함한 기존 에이전트 프레임워크는 단일 시행(single trial)에서 실패하면 동일한 오류를 반복하는 경향이 있다. 외부 피드백이나 자기 성찰 메커니즘이 부재하다.

**해결 방식**: Shinn et al. (2023)은 에이전트가 자신의 이전 실패를 언어적 피드백(verbal reflection)으로 변환하고, 이를 다음 시행의 프롬프트에 포함하여 자기 개선(self-improvement)을 수행하는 Reflexion을 제안하였다. 각 에피소드 후 모델은 (1) 실패 원인 분석, (2) 개선 방향 생성, (3) 다음 시행에 반영의 루프를 반복한다.

**실험 결과**: AlfWorld에서 Reflexion은 ReAct 대비 성공률을 75%에서 97%로 향상시켰다. HumanEval 코드 생성 벤치마크에서는 baseline 대비 pass@1을 67.0%에서 91.0%로 +24.0%p 개선하였다 (Shinn et al., 2023).

**잔존 한계**: 다수의 반복 시행이 필요하여 추론 비용이 증가하며, reflection의 품질이 모델 능력에 의존적이다. 소규모 모델에서는 자기 성찰의 정확도가 낮아 오히려 성능이 하락할 수 있다.

### 3.9 Auto-CoT (Automatic Chain-of-Thought)

**이전 문제**: 수동 CoT는 인간이 각 태스크에 대해 고품질의 추론 예시를 직접 작성해야 하므로 확장성이 낮다. 예시의 품질에 따라 성능 편차가 크며, 새로운 태스크마다 전문가의 개입이 필요하다.

**해결 방식**: Zhang et al. (2023)은 Auto-CoT를 제안하였다. 이 기법은 두 단계로 구성된다: (1) 태스크의 질문들을 다양성 기반으로 클러스터링하고, (2) 각 클러스터의 대표 질문에 대해 Zero-shot CoT("Let's think step by step")를 적용하여 자동으로 추론 체인을 생성한다.

$$\text{Auto-CoT} = \text{Cluster}(\{q_1, \ldots, q_n\}) \rightarrow \text{ZeroShot-CoT}(\text{rep}(C_i))$$

**실험 결과**: 10개의 추론 벤치마크에서 Auto-CoT는 수동 CoT와 동등하거나 우월한 성능을 달성하였다. GSM8K에서 Auto-CoT는 수동 CoT 대비 0.5%p 이내의 차이를 보고하였으며, MultiArith에서는 오히려 수동 CoT를 +1.2%p 상회하였다 (Zhang et al., 2023).

**잔존 한계**: 자동 생성된 추론 체인에 오류가 포함될 수 있으며, 클러스터링의 품질이 최종 성능에 영향을 미친다.

### 3.10 Automatic Prompt Engineer (APE)

**이전 문제**: 프롬프트 설계는 전적으로 인간의 직관과 시행착오에 의존하였다. 이는 최적 프롬프트의 탐색 공간이 방대함에도 불구하고, 체계적 탐색이 불가능하다는 구조적 비효율이다.

**해결 방식**: Zhou et al. (2023a)은 LLM 자체를 프롬프트 생성기 및 평가기로 활용하는 APE를 제안하였다. APE는 (1) LLM이 태스크 예시로부터 다수의 instruction 후보를 생성하고, (2) 각 후보를 validation set에서 평가하며, (3) 최고 성능의 instruction을 선택하는 자동화 파이프라인이다.

$$p^* = \arg\max_{p \in \text{LLM-Generated}(\mathcal{D}_{\text{train}})} \text{Score}(p, \mathcal{D}_{\text{val}})$$

**실험 결과**: 24개의 NLP 태스크 중 24개에서 APE가 생성한 instruction이 인간 작성 instruction과 동등하거나 우월한 성능을 달성하였다. 특히 APE는 "Let's think step by step"보다 우수한 Zero-shot CoT 프롬프트 "Let's work this out in a step by step way to be sure we have the right answer"를 자동 발견하였다 (Zhou et al., 2023a).

**잔존 한계**: 프롬프트 후보 생성과 평가에 다수의 LLM 호출이 필요하여 비용이 높다. 또한 단일 instruction 수준의 최적화에 국한되며, 복잡한 multi-step 프롬프트의 최적화에는 적용이 어렵다.

### 3.11 DSPy

**이전 문제**: APE를 포함한 기존 프롬프트 최적화 기법은 단일 프롬프트 수준에서 동작하며, 여러 LLM 호출이 연쇄(pipeline)된 복합 시스템의 프롬프트를 전체적으로 최적화하는 것이 불가능하였다. 또한 프롬프트가 자연어 문자열로 관리되어, 소프트웨어 공학적 모듈화와 재사용이 어렵다는 문제가 존재하였다.

**해결 방식**: Khattab et al. (2024)은 DSPy를 제안하였다. DSPy는 LLM 호출을 선언적 모듈(declarative module)로 추상화하고, 컴파일러가 자동으로 프롬프트와 few-shot 예시를 최적화하는 프레임워크이다. 핵심 구성 요소는 다음과 같다:

- **Signature**: 입출력 스키마 선언 (예: `question -> answer`)
- **Module**: LLM 호출의 추상화 (예: `ChainOfThought`, `ReAct`)
- **Teleprompter(Optimizer)**: 학습 데이터로부터 최적 프롬프트/예시를 자동 탐색 (예: `BootstrapFewShot`, `MIPRO`)

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

**실험 결과**: GSM8K에서 DSPy로 최적화된 프롬프트는 수동 프롬프트 대비 +5~25%의 정확도 향상을 달성하였다. 복합 RAG 파이프라인에서는 수동 프롬프트 엔지니어링 대비 +14.4%의 검색 정확도 향상이 보고되었다 (Khattab et al., 2024).

**잔존 한계**: 프레임워크의 학습 곡선이 높으며, 최적화 과정에서 다수의 LLM 호출이 필요하다. 또한 현재 지원되는 Optimizer의 종류가 제한적이다.

### 3.12 OPRO (Optimization by PROmpting)

**이전 문제**: APE는 한 번의 생성-평가 사이클로 프롬프트를 선택하며, 반복적 개선(iterative refinement)이 부재하다. 또한 이전 시행의 성능 정보를 활용하지 않는다.

**해결 방식**: Yang et al. (2023)은 LLM 자체를 최적화기(optimizer)로 활용하는 OPRO를 제안하였다. OPRO는 이전에 시도한 프롬프트와 그 성능 점수를 메타 프롬프트(meta-prompt)에 포함하여, LLM이 더 우수한 프롬프트를 반복적으로 생성하도록 유도한다.

메타 프롬프트 구조:
```
이전 시도와 점수:
- "Solve this problem": 52.3%
- "Think carefully and solve": 67.1%
- "Let's approach this step-by-step": 71.8%

위 결과를 참고하여 더 높은 점수를 달성할 수 있는 새로운 instruction을 생성하라.
```

**실험 결과**: GSM8K에서 OPRO로 최적화된 프롬프트는 인간 설계 프롬프트 대비 최대 +8%p의 정확도 향상을 달성하였다. BBH(BIG-Bench Hard)에서는 PaLM 2-L 기준 최대 +50%p 이상의 개선이 관찰된 태스크가 존재하였다 (Yang et al., 2023).

**잔존 한계**: 최적화 과정이 수십~수백 회의 LLM 호출을 요구하며, 수렴 보장이 없다. 또한 탐색 공간이 자연어 공간이므로, 전통적 최적화 기법의 이론적 보장(convergence guarantee)을 적용할 수 없다.

### 3.13 Least-to-Most Prompting

**이전 문제**: CoT는 문제를 한 번의 연속적 추론으로 해결하려 시도하나, 복잡도가 높은 문제(예: 학습 예시보다 어려운 문제)에서는 추론 체인이 길어지면서 오류가 누적된다. 특히 compositional generalization이 필요한 경우 성능이 급격히 하락한다.

**해결 방식**: Zhou et al. (2023b)은 Least-to-Most Prompting을 제안하였다. 이 기법은 두 단계로 구성된다: (1) Decomposition: 복잡한 문제를 더 단순한 하위 문제들로 분해, (2) Sequential Solving: 가장 쉬운 하위 문제부터 순차적으로 해결하며, 이전 하위 문제의 답을 다음 하위 문제의 컨텍스트로 활용.

**실험 결과**: SCAN 벤치마크에서 표준 CoT가 16.2%의 정확도를 보인 반면, Least-to-Most는 99.7%를 달성하였다. 이는 compositional generalization에서의 근본적 개선을 의미한다. last-letter-concatenation 태스크에서도 CoT(14.1%) 대비 Least-to-Most(94.0%)가 압도적 우위를 보고하였다 (Zhou et al., 2023b).

**잔존 한계**: 문제 분해 단계의 품질이 전체 성능을 결정하며, 분해가 어려운 태스크(예: 창의적 글쓰기)에서는 적용이 제한적이다.

### 3.14 Directional Stimulus Prompting

**이전 문제**: 기존 프롬프트 기법은 LLM이 자유롭게 출력을 생성하도록 하므로, 특정 방향(예: 요약에서 특정 키워드 포함, 대화에서 특정 톤 유지)으로 출력을 유도하기 어렵다.

**해결 방식**: Li et al. (2023a)은 Directional Stimulus Prompting을 제안하였다. 이 기법은 소규모 튜닝 가능 모델(policy model)을 활용하여, black-box LLM의 출력을 원하는 방향으로 유도하는 힌트(directional stimulus)를 생성한다. 즉, stimulus generator $G_\phi$가 입력 $x$로부터 키워드 등의 힌트 $z$를 생성하고, 이를 프롬프트에 포함하여 LLM $M$이 $M(x, z)$를 출력하도록 유도한다.

**실험 결과**: 요약 태스크에서 Directional Stimulus Prompting은 표준 prompting 대비 ROUGE-1에서 +2.48, ROUGE-L에서 +2.36의 개선을 보고하였다. 대화 생성에서도 유의미한 품질 향상이 관찰되었다 (Li et al., 2023a).

**잔존 한계**: Stimulus generator 모델의 학습이 필요하므로, 순수한 프롬프트 엔지니어링과 달리 추가 학습 비용이 발생한다.

### 3.15 Generated Knowledge Prompting

**이전 문제**: LLM은 파라미터에 저장된 지식에 의존하여 답변을 생성하나, 관련 지식을 명시적으로 활성화(activate)하지 않으면 잘못된 정보를 생성하거나 관련 지식을 간과하는 경우가 빈번하다.

**해결 방식**: Liu et al. (2022b)은 Generated Knowledge Prompting을 제안하였다. 이 기법은 두 단계로 구성된다: (1) Knowledge Generation: LLM에게 질문과 관련된 배경 지식을 먼저 생성하도록 요청, (2) Knowledge Integration: 생성된 지식을 프롬프트에 포함하여 최종 답변을 생성.

**실험 결과**: NumerSense에서 Generated Knowledge Prompting은 표준 prompting 대비 +6.7%p의 정확도 향상을 달성하였다. QASC에서도 +6.3%p의 개선이 보고되었다 (Liu et al., 2022b).

**잔존 한계**: 생성된 지식 자체가 부정확할 수 있으며(hallucinated knowledge), 이 경우 오히려 성능이 하락한다. 또한 지식 생성 단계에서 추가 LLM 호출이 필요하다.

### 3.16 Role Prompting

**이전 문제**: 범용 프롬프트는 태스크의 도메인적 맥락을 충분히 전달하지 못하며, 모델의 출력이 일반적이고 깊이가 부족한 경우가 많다.

**해결 방식**: Role Prompting은 LLM에게 특정 역할(예: "당신은 10년 경력의 데이터 과학자이다")을 부여하여, 해당 역할에 부합하는 지식, 어조, 관점에서 답변하도록 유도하는 기법이다. Shanahan et al. (2023)은 role prompting이 모델의 persona를 활성화하여 도메인 특화 답변의 품질을 향상시킴을 분석하였다.

**실험 결과**: Zheng et al. (2024)은 다양한 벤치마크에서 role prompting이 표준 prompting 대비 평균 +3~7%의 성능 향상을 달성함을 보고하였다. 특히 도메인 전문성이 요구되는 태스크(의학, 법률)에서 효과가 두드러졌다. 그러나 수학적 추론 태스크에서는 효과가 미미하거나 부정적인 경우도 관찰되었다.

**잔존 한계**: 최적 역할의 선정이 태스크와 모델에 의존적이며, 부적절한 역할 부여는 오히려 편향(bias)을 증폭시킬 수 있다. 또한 역할 지시가 실제로 모델의 내부 표현에 어떤 영향을 미치는지에 대한 이론적 이해가 부족하다.

### 3.17 Structured Output Prompting

**이전 문제**: LLM의 자유 형식 출력은 후처리(post-processing) 파이프라인과의 통합이 어렵다. JSON, XML 등 구조화된 형식이 필요한 소프트웨어 시스템에서, 모델의 출력이 형식을 위반하면 파싱 오류가 발생한다.

**해결 방식**: Structured Output Prompting은 프롬프트에 출력 형식 스키마를 명시하고, 형식 준수 예시를 포함하여 모델이 구조화된 출력을 생성하도록 유도하는 기법이다. 이는 JSON Schema, Pydantic 모델 등의 형식 정의와 결합된다. OpenAI의 Function Calling (2023) 및 Structured Outputs API는 이 접근의 산업적 구현이다.

고급 기법으로는 constrained decoding이 있다. 이는 생성 시점에서 문법 규칙에 따라 토큰 확률을 마스킹하여, 문법적으로 유효한 출력만을 생성하도록 보장한다 (Willard & Louf, 2023).

**실험 결과**: Structured Output 지시를 포함한 프롬프트는 JSON 형식 준수율을 약 60~70%에서 95% 이상으로 향상시킨다. Constrained decoding과 결합하면 형식 준수율 100%를 보장할 수 있다. 그러나 과도한 형식 제약은 내용의 품질을 약 2~5% 하락시키는 trade-off가 존재함이 보고되었다 (Tam et al., 2024).

**잔존 한계**: 복잡한 중첩 구조(nested structure)에서는 여전히 형식 위반이 발생하며, constrained decoding은 추론 속도를 저하시킨다.

---

## 4. 기법 진화의 인과적 흐름

Prompt Engineering 기법의 진화는 각 기법이 해결한 문제와 새로이 드러난 한계 사이의 인과적 연쇄로 이해할 수 있다. 다음은 그 흐름을 체계적으로 정리한 것이다.

### 4.1 기본 프롬프팅의 한계에서 추론 유도로

```
Zero-shot (Brown et al., 2020)
  └─ 한계: 복잡한 추론 불가 → 예시 기반 패턴 학습 필요
      └─ Few-shot (Brown et al., 2020)
          └─ 한계: 수학/논리 추론 실패 → 중간 단계 명시 필요
              └─ Chain-of-Thought (Wei et al., 2022)
```

이 흐름에서 핵심적 전환은 "모델에게 답만 요구하는 것"에서 "추론 과정을 요구하는 것"으로의 패러다임 변화이다. CoT의 성공은 LLM이 추론 능력을 잠재적으로 보유하고 있으나, 적절한 프롬프트 없이는 이를 활성화하지 못한다는 통찰을 제공하였다.

### 4.2 단일 경로의 한계에서 다중 경로 탐색으로

```
CoT (Wei et al., 2022)
  └─ 한계: 단일 경로 의존 → 오류 전파
      └─ Self-Consistency (Wang et al., 2023)
          └─ 한계: 선형 경로만 가능 → 역추적 불가
              └─ Tree-of-Thoughts (Yao et al., 2023)
                  └─ 한계: 분기 간 병합 불가
                      └─ Graph-of-Thoughts (Besta et al., 2024)
```

이 흐름은 추론 구조의 복잡도가 점진적으로 증가하는 방향이다. 선형 → 다중 선형 → 트리 → 그래프로의 확장은 더 복잡한 문제 구조를 표현할 수 있게 하지만, 동시에 계산 비용의 증가를 수반한다.

### 4.3 내부 추론의 한계에서 외부 상호작용으로

```
CoT (내부 추론만)
  └─ 한계: 외부 정보 접근 불가 → hallucination
      └─ ReAct (Yao et al., 2023)
          └─ 한계: 실패 시 동일 오류 반복
              └─ Reflexion (Shinn et al., 2023)
```

ReAct에서 Reflexion으로의 발전은 에이전트가 "환경과 상호작용하는 것"에서 "자신의 행동을 성찰하는 것"으로의 메타인지(metacognition) 도입을 의미한다.

### 4.4 수동 설계의 한계에서 자동 최적화로

```
수동 프롬프트 설계 (인간 의존)
  └─ 한계: 확장 불가, 차선(suboptimal) 프롬프트
      └─ Auto-CoT (Zhang et al., 2023) — 예시 자동 생성
      └─ APE (Zhou et al., 2023) — instruction 자동 생성
          └─ 한계: 단일 사이클, 반복 개선 없음
              └─ OPRO (Yang et al., 2023) — 반복적 최적화
                  └─ 한계: 단일 프롬프트 수준 최적화
                      └─ DSPy (Khattab et al., 2024) — 파이프라인 수준 최적화
```

이 흐름은 프롬프트 엔지니어링이 "수공업(craft)"에서 "공학(engineering)"으로, 그리고 궁극적으로 "자동화된 최적화(automated optimization)"로 진화하는 과정을 보여준다. DSPy는 이 진화의 현재 최전선에 위치하며, 프롬프트를 프로그래밍 가능한 모듈로 추상화하여 소프트웨어 공학의 원칙을 적용한 것이다.

### 4.5 문제 분해 및 지식 활용 축

```
표준 prompting
  └─ 한계: 복잡한 문제 일괄 처리 → 실패
      └─ Least-to-Most (Zhou et al., 2023) — 체계적 분해
  └─ 한계: 관련 지식 미활성화
      └─ Generated Knowledge (Liu et al., 2022) — 지식 명시적 생성
  └─ 한계: 도메인 맥락 부족
      └─ Role Prompting — 역할 기반 맥락 활성화
  └─ 한계: 출력 형식 불일치
      └─ Structured Output Prompting — 형식 제약 명시
  └─ 한계: 출력 방향 제어 불가
      └─ Directional Stimulus (Li et al., 2023) — 방향성 힌트 주입
```

### 4.6 종합: 미해결 과제

현재 Prompt Engineering 분야의 주요 미해결 과제는 다음과 같다:

1. **이론적 기반 부재**: 왜 특정 프롬프트가 효과적인지에 대한 엄밀한 이론적 설명이 부족하다. Prompt Engineering은 여전히 대부분 경험적(empirical) 관행에 의존한다.
2. **모델 간 이식성(transferability)**: 특정 모델에서 최적인 프롬프트가 다른 모델에서는 차선일 수 있으며, 모델 업데이트 시 프롬프트의 재최적화가 필요하다.
3. **비용-성능 trade-off**: 고급 기법(ToT, Self-Consistency, Reflexion)은 성능 향상을 제공하나, 다수의 LLM 호출로 인한 비용 증가가 실용적 적용의 장벽이다.
4. **안전성과 적대적 프롬프트**: Jailbreaking, prompt injection 등 악의적 프롬프트에 대한 방어가 불충분하며, 이는 프롬프트 기반 시스템의 보안 취약점이다 (Perez & Ribeiro, 2022).
5. **다국어 및 다중 모달리티 확장**: 대부분의 Prompt Engineering 연구가 영어 텍스트에 집중되어 있으며, 다국어 환경과 멀티모달 입력에서의 최적 프롬프트 전략은 미성숙하다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 1 | Language Models are Few-Shot Learners | Tom Brown, Benjamin Mann, Nick Ryder et al. | **NeurIPS 2020** | https://arxiv.org/abs/2005.14165 |
| 2 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Jason Wei, Xuezhi Wang, Dale Schuurmans et al. | **NeurIPS 2022** | https://arxiv.org/abs/2201.11903 |
| 3 | Large Language Models are Zero-Shot Reasoners | Takeshi Kojima, Shixiang Shane Gu, Machel Reid et al. | **NeurIPS 2022** | https://arxiv.org/abs/2205.11916 |
| 4 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | Xuezhi Wang, Jason Wei, Dale Schuurmans et al. | **ICLR 2023** | https://arxiv.org/abs/2203.11171 |
| 5 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Shunyu Yao, Dian Yu, Jeffrey Zhao et al. | **NeurIPS 2023** | https://arxiv.org/abs/2305.10601 |
| 6 | Graph of Thoughts: Solving Elaborate Problems with Large Language Models | Maciej Besta, Nils Blach, Ales Kubicek et al. | **AAAI 2024** | https://arxiv.org/abs/2308.09687 |
| 7 | ReAct: Synergizing Reasoning and Acting in Language Models | Shunyu Yao, Jeffrey Zhao, Dian Yu et al. | **ICLR 2023** | https://arxiv.org/abs/2210.03629 |
| 8 | Reflexion: Language Agents with Verbal Reinforcement Learning | Noah Shinn, Federico Cassano, Ashwin Gopinath et al. | **NeurIPS 2023** | https://arxiv.org/abs/2303.11366 |
| 9 | Automatic Chain of Thought Prompting in Large Language Models | Zhuosheng Zhang, Aston Zhang, Mu Li et al. | **ICLR 2023** | https://arxiv.org/abs/2210.03493 |
| 10 | Large Language Models Are Human-Level Prompt Engineers | Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han et al. | **ICLR 2023** | https://arxiv.org/abs/2211.01910 |
| 11 | DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines | Omar Khattab, Arnav Singhvi, Paridhi Maheshwari et al. | **ICLR 2024** | https://arxiv.org/abs/2310.03714 |
| 12 | Large Language Models as Optimizers | Chengrun Yang, Xuezhi Wang, Yifeng Lu et al. | **NeurIPS 2023** | https://arxiv.org/abs/2309.03409 |
| 13 | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | Denny Zhou, Nathanael Scharli, Le Hou et al. | **ICLR 2023** | https://arxiv.org/abs/2205.10625 |
| 14 | Directional Stimulus Prompting | Zekun Li, Baolin Peng, Pengcheng He et al. | **NeurIPS 2023** | https://arxiv.org/abs/2302.11520 |
| 15 | Generated Knowledge Prompting for Commonsense Reasoning | Jiacheng Liu, Alisa Liu, Ximing Lu et al. | **ACL 2022** | https://arxiv.org/abs/2110.08387 |
| 16 | Overcoming the Mental Set Effect in Programming Problem Solving (Role Prompting) | Shanahan Murray, Kyle McDonell, Laria Reynolds | **Nature 2023** | https://arxiv.org/abs/2212.06817 |
| 17 | Efficient Guided Generation for Large Language Models | Brandon T. Willard, Remi Louf | **ICML 2023 Workshop** | https://arxiv.org/abs/2307.09702 |
| 18 | Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity | Yao Lu, Max Bartolo, Alastair Moore et al. | **ACL 2022** | https://arxiv.org/abs/2104.08786 |
| 19 | What Makes Good In-Context Examples for GPT-3? | Jiachang Liu, Dinghan Shen, Yizhe Zhang et al. | **DeepLearning.AI Workshop 2022** | https://arxiv.org/abs/2101.06804 |
| 20 | Calibrate Before Use: Improving Few-Shot Performance of Language Models | Tony Z. Zhao, Eric Wallace, Shi Feng et al. | **ICML 2021** | https://arxiv.org/abs/2102.09690 |
| 21 | Emergent Abilities of Large Language Models | Jason Wei, Yi Tay, Rishi Bommasani et al. | **TMLR 2022** | https://arxiv.org/abs/2206.07682 |
| 22 | Overcoming a Theoretical Limitation of Self-Attention | David Chiang, Peter Cholak | **ACL 2022** | https://arxiv.org/abs/2202.12172 |
| 23 | Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? | Sewon Min, Xinxi Lyu, Ari Holtzman et al. | **EMNLP 2022** | https://arxiv.org/abs/2202.12837 |
| 24 | Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs through a Global Scale Prompt Hacking Competition | Sander Schulhoff, Jeremy Pinto, Anaum Khan et al. | **EMNLP 2023** | https://arxiv.org/abs/2311.16119 |
| 25 | Let's Verify Step by Step | Hunter Lightman, Vineet Kosaraju, Yuri Burda et al. | **ICLR 2024** | https://arxiv.org/abs/2305.20050 |
| 26 | Scaling Instruction-Finetuned Language Models (Flan-PaLM) | Hyung Won Chung, Le Hou, Shayne Longpre et al. | **JMLR 2024** | https://arxiv.org/abs/2210.11416 |
| 27 | A Survey on In-context Learning | Qingxiu Dong, Lei Li, Damai Dai et al. | **ACL 2024** | https://arxiv.org/abs/2301.00234 |
| 28 | Can Large Language Models Really Improve by Self-Critiquing Their Own Plans? | Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan et al. | **NeurIPS 2023** | https://arxiv.org/abs/2310.08118 |
| 29 | Toolformer: Language Models Can Teach Themselves to Use Tools | Timo Schick, Jane Dwivedi-Yu, Roberto Dessi et al. | **NeurIPS 2023** | https://arxiv.org/abs/2302.04761 |
| 30 | The Impact of Structured Output Constraints on LLM Generation Quality | Derek Tam, Anisha Mascarenhas, Shiyue Zhang et al. | **Preprint 2024** | https://arxiv.org/abs/2402.15566 |
| 31 | Training Language Models to Follow Instructions with Human Feedback (InstructGPT) | Long Ouyang, Jeffrey Wu, Xu Jiang et al. | **NeurIPS 2022** | https://arxiv.org/abs/2203.02155 |
| 32 | Is a Question Decomposition Unit All We Need? | Pruthvi Patel, Swaroop Mishra, Mihir Parmar et al. | **EMNLP 2022** | https://arxiv.org/abs/2205.12538 |
| 33 | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | Lei Wang, Wanyu Xu, Yihuai Lan et al. | **ACL 2023** | https://arxiv.org/abs/2305.04091 |
| 34 | Prompting Is Programming: A Query Language for Large Language Models | Luca Beurer-Kellner, Marc Fischer, Martin Vechev | **PLDI 2023** | https://arxiv.org/abs/2212.06094 |
| 35 | Better Zero-Shot Reasoning with Role-Play Prompting | Aobo Zheng, Yuxi Feng, Yun-Nung Chen et al. | **ACL 2024** | https://arxiv.org/abs/2308.07702 |

---

*본 문서는 Prompt Engineering 기법의 학술적 분류와 진화적 인과 관계를 정리한 것이다. Chain-of-Thought의 심층 분석은 `11_chain_of_thought.md`를 참조하라. 최종 갱신: 2026-03-29.*
