# Chain-of-Thought Reasoning과 Reasoning Models

## 1. 기법의 정의

Chain-of-Thought(CoT) 추론이란, 대규모 언어 모델(LLM)이 최종 답변을 생성하기 전에 중간 추론 단계(intermediate reasoning steps)를 명시적으로 생성하도록 유도하는 기법이다. 전통적인 direct prompting에서는 입력 x가 주어지면 모델이 곧바로 출력 y를 생성하는 p(y|x) 형태였으나, CoT prompting에서는 중간 추론 체인 z₁, z₂, ..., zₙ을 순차적으로 생성한 뒤 최종 답변 y를 도출하는 p(z₁, ..., zₙ, y|x) 형태로 전환하는 것이다.

이 기법의 핵심 원리는 복잡한 추론 문제를 여러 개의 단순한 하위 문제(sub-problem)로 분해(decomposition)하여 각 단계에서의 오류 확률을 낮추는 것이다. 이는 인간의 시스템 2 사고(System 2 thinking, Kahneman 2011)에 해당하며, 느리지만 의도적인(deliberate) 추론 과정을 모사하는 것이다.

Reasoning Model은 CoT의 확장으로, 모델 자체가 학습 과정에서 내재적 추론 능력을 획득하도록 강화학습(RL)을 적용한 모델 계열이다. OpenAI의 o1(2024), o3(2025), DeepSeek-R1(2025), Alibaba의 QwQ(2024) 등이 대표적이며, 이들은 test-time compute scaling이라는 새로운 패러다임을 정립한 것이다.

---

## 2. 기존 기법의 한계와 CoT 등장 배경

### 2.1 Direct Prompting의 구조적 실패

Standard prompting(direct prompting)은 모델에 질문을 제시하고 곧바로 답변을 요구하는 방식이다. 이 방식은 단일 단계 지식 검색(factoid QA)에서는 충분한 성능을 보이나, 다단계 산술 추론(multi-step arithmetic reasoning), 상식 추론(commonsense reasoning), 기호 추론(symbolic reasoning) 등에서 체계적으로 실패하는 것이다.

**문제의 본질:** LLM의 Transformer 아키텍처는 고정된 깊이의 연산 그래프(computation graph)를 가진다. 즉, 입력 토큰으로부터 출력 토큰까지의 연산 단계 수가 모델의 레이어 수로 제한되는 것이다. 이는 Feng et al. (2024)이 형식적으로 증명한 바와 같이, 고정 깊이의 Transformer는 inherently serial한 문제(예: 다단계 산술)를 해결할 수 없다는 것을 의미하는 것이다. CoT는 이 문제를 중간 토큰 생성을 통해 우회하는 방법이다 — 각 추론 단계를 별도의 토큰으로 출력함으로써, 사실상 모델의 연산 깊이를 동적으로 확장하는 효과를 가지는 것이다.

**실증적 근거:** PaLM 540B 모델에서 GSM8K(초등 수학 문제) 벤치마크의 direct prompting 정확도는 17.9%에 불과하였으나, CoT prompting 적용 시 58.1%로 3.2배 향상된 것이다 (Wei et al., 2022).

### 2.2 Scaling만으로는 불충분

모델 크기를 증가시키는 것(parameter scaling)만으로는 추론 문제를 해결할 수 없다는 점이 CoT 등장의 핵심 동기이다. GPT-3 175B, PaLM 540B 등 당시 최대 규모 모델도 direct prompting 하에서 GSM8K 정확도가 20% 미만이었으며, 이는 단순한 초등학교 수준의 산술 문제임을 감안하면 심각한 한계인 것이다. 이 관찰은 "모델을 키우는 것"에서 "추론 방식을 개선하는 것"으로 연구 방향의 전환을 촉발한 것이다.

---

## 3. 주요 기법 상세

### 3.1 Few-shot Chain-of-Thought Prompting

**이전 문제:** Direct prompting으로는 다단계 추론 문제 해결이 불가능하다.

**해결:** Wei et al. (2022)은 프롬프트에 질문-추론과정-답변의 예시(exemplar)를 포함시키는 few-shot CoT prompting을 제안한 것이다. 모델은 주어진 예시의 추론 형식을 모방하여 새로운 문제에 대해서도 중간 추론 단계를 생성하게 되는 것이다.

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11. The answer is 11.

Q: [새로운 문제]
A: [모델이 유사한 형식으로 추론 과정을 생성]
```

**벤치마크 결과 (PaLM 540B):**

| 벤치마크 | Direct Prompting | Few-shot CoT | 향상폭 |
|----------|-----------------|--------------|--------|
| GSM8K | 17.9% | 58.1% | +40.2%p |
| SVAMP | 79.0% | 86.6% | +7.6%p |
| MAWPS | 84.7% | 93.3% | +8.6%p |
| AQuA | 25.2% | 35.8% | +10.6%p |

**새로운 문제:** (1) 예시(exemplar) 설계에 대한 인간 전문가의 수동 작업이 필요하다. (2) 예시의 품질과 구성에 따라 성능이 크게 변동한다. (3) 약 100B 파라미터 이하의 소형 모델에서는 CoT가 오히려 성능을 저하시키는 현상(inverse scaling)이 관찰되는 것이다. Wei et al.은 이를 "emergent ability"로 해석하였으나, 후속 연구에서 이 해석에 대한 논쟁이 지속되고 있는 것이다.

### 3.2 Zero-shot Chain-of-Thought

**이전 문제:** Few-shot CoT는 태스크별로 수동으로 예시를 작성해야 하므로 확장성이 제한된다.

**해결:** Kojima et al. (2022)은 프롬프트 끝에 "Let's think step by step"이라는 단일 문구만 추가하면 모델이 자발적으로 추론 과정을 생성한다는 것을 발견한 것이다. 이는 LLM의 사전학습 데이터에 포함된 단계적 설명(step-by-step explanation) 패턴을 활성화하는 것으로 해석되는 것이다.

**벤치마크 결과 (InstructGPT 175B):**
- MultiArith: Direct 17.7% → Zero-shot CoT 78.7%
- GSM8K: Direct 12.5% → Zero-shot CoT 40.7%
- SVAMP: Direct 63.7% → Zero-shot CoT 70.8%

**새로운 문제:** (1) Few-shot CoT 대비 성능이 일관적으로 낮다. (2) 트리거 문구의 선택에 따라 성능 편차가 크다 ("Let's think step by step"이 최적이나 이를 사전에 알 수 없다). (3) 모델이 생성하는 추론 과정의 충실성(faithfulness)이 보장되지 않는 것이다.

### 3.3 Self-Consistency

**이전 문제:** CoT는 단일 추론 경로(single reasoning path)에 의존하므로, 해당 경로에 오류가 포함되면 최종 답변도 오류가 되는 것이다.

**해결:** Wang et al. (2023)은 동일 문제에 대해 temperature sampling으로 다수의 추론 경로를 생성한 뒤, 최종 답변에 대한 다수결 투표(majority voting)를 수행하는 self-consistency 기법을 제안한 것이다. 이는 추론 문제에는 정답에 도달하는 경로가 여러 개 존재한다는 직관에 기반하며, 올바른 답변으로 수렴하는 경로가 오답으로 수렴하는 경로보다 많을 것이라는 가정 하에 작동하는 것이다.

**벤치마크 결과 (PaLM 540B, 40 paths):**

| 벤치마크 | CoT (greedy) | Self-Consistency | 향상폭 |
|----------|-------------|-----------------|--------|
| GSM8K | 56.5% | 74.4% | +17.9%p |
| SVAMP | 79.0% | 90.0% | +11.0%p |
| AQuA | 35.8% | 48.0% | +12.2%p |

**새로운 문제:** (1) N개의 경로를 생성하므로 추론 비용이 N배로 증가한다 (N=40일 경우 40배). (2) 다수결 투표는 최종 답변만 비교하므로, 추론 과정의 품질을 평가하지 못한다. (3) 모든 경로가 동일한 오류 패턴을 공유하면 다수결도 실패하는 것이다.

### 3.4 Automatic Chain-of-Thought (Auto-CoT)

**이전 문제:** Few-shot CoT의 예시를 수동으로 설계하는 것은 비용이 높고, 예시 선택에 따른 성능 편차가 크다.

**해결:** Zhang et al. (2023)은 Auto-CoT를 제안하여, 질문 클러스터링과 zero-shot CoT 생성을 결합함으로써 예시를 자동으로 구성하는 것이다. 구체적으로, (1) 태스크의 질문들을 문장 임베딩 기반으로 클러스터링하고, (2) 각 클러스터의 대표 질문에 대해 zero-shot CoT로 추론 과정을 생성하며, (3) 생성된 질문-추론-답변 쌍을 few-shot exemplar로 사용하는 것이다.

**새로운 문제:** 자동 생성된 추론 과정에 오류가 포함될 수 있으며, 이러한 오류가 전파되어 후속 문제의 추론에도 영향을 미칠 수 있는 것이다.

### 3.5 Complexity-Based CoT

**이전 문제:** Self-consistency의 다수결 투표는 모든 추론 경로를 동등하게 취급하나, 복잡한 문제일수록 더 긴(더 상세한) 추론 과정이 필요하다.

**해결:** Fu et al. (2023)은 complexity-based prompting을 제안한 것이다. 핵심 발견은, 더 많은 추론 단계를 포함하는 경로(complex chains)가 더 높은 정확도를 보인다는 것이다. 따라서 (1) 다수의 경로 생성 후, (2) 추론 단계 수가 임계값 이상인 경로만 필터링하고, (3) 해당 경로들에 대해 다수결 투표를 수행하는 것이다.

**벤치마크 결과:** GSM8K에서 self-consistency 대비 +2-3%p 추가 향상이 관찰된 것이다.

**새로운 문제:** 추론 단계 수가 많다고 해서 반드시 품질이 높은 것은 아니며, 불필요하게 긴 추론(verbose reasoning)이 오히려 오류를 유발할 수 있는 것이다.

### 3.6 Tree of Thoughts (ToT)

**이전 문제:** CoT와 self-consistency는 모두 선형적(linear) 추론 경로를 사용하므로, 중간 단계에서 잘못된 방향으로 진행되면 되돌릴(backtrack) 수 없는 것이다.

**해결:** Yao et al. (2023)은 Tree of Thoughts(ToT)를 제안하여, 각 추론 단계를 트리의 노드로 모델링하고, BFS(Breadth-First Search) 또는 DFS(Depth-First Search)를 통해 탐색 공간을 체계적으로 탐색하는 것이다. 각 노드에서 LLM이 해당 부분 해(partial solution)의 유망성(promise)을 평가하고, 유망하지 않은 경로는 가지치기(pruning)하는 것이다.

```
           [문제]
          /      \
     [단계1a]   [단계1b]
      /   \        |
  [2a]   [2b]    [2c]    ← 각 노드에서 LLM이 평가(evaluate)
   |       ✗       |
  [3a]            [3c]    ← 유망한 경로만 확장
   |               |
 [답a]           [답c]   ← 최종 답변 후보
```

**벤치마크 결과:**
- Game of 24: CoT 4.0% → ToT 74.0% (+70.0%p)
- Creative Writing (일관성 점수): CoT 6.19 → ToT 7.56
- Mini Crosswords: CoT 15.6% → ToT 60.0%

**새로운 문제:** (1) 각 노드에서 LLM 호출이 필요하므로 연산 비용이 극도로 높다 (단일 문제에 수십~수백 회의 LLM 호출). (2) 평가 함수(evaluation function)의 설계가 태스크 의존적이다. (3) 단순한 추론 문제에서는 CoT + self-consistency 대비 이점이 없는 것이다.

### 3.7 Graph of Thoughts (GoT)

**이전 문제:** ToT는 트리 구조이므로 서로 다른 분기(branch)의 부분 해를 결합(aggregate)할 수 없다.

**해결:** Besta et al. (2024)은 Graph of Thoughts(GoT)를 제안하여, 추론 과정을 임의의 방향성 비순환 그래프(DAG)로 모델링한 것이다. 이를 통해 (1) 여러 사고의 병합(aggregation), (2) 사고의 정제(refinement), (3) 복수 경로의 결합이 가능해진 것이다.

**새로운 문제:** 그래프 구조의 설계와 탐색 전략이 더욱 복잡해지며, 실제 적용 가능한 태스크가 제한적인 것이다.

### 3.8 Faithful Chain-of-Thought

**이전 문제:** 표준 CoT에서 모델이 생성하는 추론 과정이 실제 내부 연산과 일치하지 않을 수 있다 — 즉, 추론 과정이 불충실(unfaithful)할 수 있는 것이다. Turpin et al. (2024)은 CoT가 biased features에 의해 체계적으로 왜곡될 수 있음을 실증한 것이다.

**해결:** Lyu et al. (2023)은 Faithful CoT를 제안하여, 자연어 추론 과정을 형식 언어(Python, Sympy 등)로의 번역과 결합한 것이다. 구체적으로, (1) 모델이 자연어로 문제를 분해하고, (2) 각 단계를 실행 가능한 코드로 변환하며, (3) 외부 인터프리터가 해당 코드를 실행하여 결과를 산출하는 것이다. 이를 통해 추론 과정의 검증 가능성(verifiability)이 확보되는 것이다.

**새로운 문제:** 모든 추론 문제가 형식 언어로 표현 가능한 것은 아니며, 상식 추론 등에서는 적용이 제한적인 것이다.

### 3.9 STaR (Self-Taught Reasoner)

**이전 문제:** CoT 능력은 프롬프팅에만 의존하므로, 모델의 내재적 추론 능력을 향상시키지 못한다. 또한 대규모 추론 과정 데이터셋을 수동으로 구축하는 것은 비용이 극히 높은 것이다.

**해결:** Zelikman et al. (2022)은 STaR를 제안하여, 모델이 스스로 추론 데이터를 생성하고 이를 통해 자기 개선(self-improvement)하는 부트스트래핑 방법을 제시한 것이다. 알고리즘은 다음과 같다:

1. 모델이 각 문제에 대해 추론 과정(rationale)과 답변을 생성한다.
2. 정답을 맞힌 경우의 추론 과정만 수집한다.
3. 틀린 문제에 대해서는 정답을 힌트(hint)로 제공하여 추론 과정을 재생성한다 (rationalization).
4. 수집된 추론 과정으로 모델을 fine-tuning한다.
5. 위 과정을 반복(iterate)한다.

**벤치마크 결과:** CommonsenseQA에서 few-shot CoT 대비 정확도를 72.5% → 80.1%로 향상시킨 것이다.

**새로운 문제:** (1) 정답 레이블이 필요하므로 완전한 비지도 방식이 아니다. (2) 틀린 추론으로 정답에 도달하는 경우(false positive)가 학습 데이터에 포함될 수 있다. (3) 부트스트래핑의 수렴 보장이 이론적으로 부재하는 것이다.

### 3.10 Quiet-STaR

**이전 문제:** STaR은 질문-답변 형식의 태스크에만 적용 가능하며, 범용적인 추론 능력 향상에는 한계가 있는 것이다.

**해결:** Zelikman et al. (2024)은 Quiet-STaR를 제안하여, 모델이 일반 텍스트의 모든 토큰 위치에서 내부적으로 "사고(thought)"를 생성하도록 학습시키는 것이다. 핵심은 각 토큰 생성 전에 <|startofthought|> 토큰을 삽입하여 내부 추론을 수행하고, 이 추론이 다음 토큰 예측에 도움이 되는 경우에만 REINFORCE 알고리즘으로 강화하는 것이다.

**기술적 세부사항:**
- Meta-tokens: `<|startofthought|>`와 `<|endofthought|>` 사이에 사고 토큰을 생성한다.
- Mixing head: 사고를 포함한 hidden state와 포함하지 않은 hidden state를 학습 가능한 가중치로 결합한다.
- Parallel generation: 모든 토큰 위치에서 동시에 사고를 생성하여 효율성을 확보한다.

**벤치마크 결과:** Mistral 7B 기준 GSM8K 5.9% → 10.9%, CommonsenseQA에서도 유의미한 향상이 관찰된 것이다.

**새로운 문제:** (1) 학습 비용이 매우 높다. (2) 소형 모델에서의 향상폭이 제한적이다. (3) 사고 토큰의 해석 가능성이 낮은 것이다.

### 3.11 Process Reward Model (PRM) vs Outcome Reward Model (ORM)

**이전 문제:** Self-consistency의 다수결 투표는 추론 과정의 품질을 평가하지 않으며, 최종 답변만으로 경로를 선택하는 것이다.

**해결:** Lightman et al. (2024)은 "Let's Verify Step by Step" 논문에서, 추론의 각 단계를 개별적으로 평가하는 Process Reward Model(PRM)이 최종 답변만 평가하는 Outcome Reward Model(ORM)보다 우수함을 대규모 실험으로 입증한 것이다.

| 특성 | ORM | PRM |
|------|-----|-----|
| **평가 대상** | 최종 답변의 정오 | 각 추론 단계의 정오 |
| **레이블링** | 최종 답변만 자동 검증 | 각 단계에 대한 인간 레이블링 필요 |
| **장점** | 데이터 구축 비용 저렴 | 추론 과정의 신뢰성 보장, 오류 조기 탐지 |
| **단점** | 잘못된 추론으로 정답에 도달하는 경우 학습 | 레이블링 비용 극히 높음 (800K 단계 레이블) |
| **MATH 정확도** (best-of-1860) | 72.4% | **78.2%** |

OpenAI는 PRM800K 데이터셋(800,000개의 단계별 레이블)을 공개하였으며, 이는 이후 Reasoning Model 개발의 핵심 기반이 된 것이다.

**새로운 문제:** (1) 인간 레이블링의 확장성 한계 — 수학 외 도메인에서의 적용이 어렵다. (2) PRM의 학습 자체에 대규모 데이터가 필요하다. (3) 단계의 정의(granularity)가 모호한 것이다.

### 3.12 Monte Carlo Tree Search (MCTS) for Reasoning

**이전 문제:** ToT의 탐색 전략(BFS/DFS)은 단순하며, PRM의 단계별 평가를 체계적으로 활용하지 못하는 것이다.

**해결:** 여러 연구 그룹이 AlphaGo 스타일의 MCTS를 LLM 추론에 적용한 것이다. 대표적으로 Feng et al. (2024)의 AlphaZero-like tree-search와 Xie et al. (2024)의 rStar 등이 있으며, (1) 각 추론 단계를 MCTS의 노드로 모델링하고, (2) PRM을 value function으로 사용하여 UCB(Upper Confidence Bound) 기반 탐색을 수행하며, (3) rollout을 통해 노드의 가치를 추정하는 것이다.

**벤치마크 결과:** rStar는 GSM8K에서 SLM(Small Language Model, 1-7B) 기반으로도 90%+ 정확도를 달성하며, 이는 기존 CoT + self-consistency 대비 15%p 이상 향상된 것이다.

**새로운 문제:** (1) 단일 문제당 수천 회의 LLM 호출이 필요하여 실시간 적용이 불가능하다. (2) MCTS의 하이퍼파라미터(탐색 폭, 깊이, UCB 상수) 조정이 필요하다. (3) 탐색 공간이 자연어이므로 분기 수(branching factor)가 매우 높은 것이다.

### 3.13 OpenAI o1

**이전 문제:** CoT prompting과 외부 탐색(ToT, MCTS)은 추론 시 추가 엔지니어링이 필요하며, 모델 자체의 내재적 추론 능력을 향상시키지 못하는 것이다.

**해결:** OpenAI (2024)는 o1 모델을 공개하였으며, 이는 대규모 강화학습을 통해 모델 내부에서 긴 사고 체인(chain of thought)을 자율적으로 생성하도록 학습된 최초의 상업적 Reasoning Model인 것이다. o1은 답변 전에 내부 "thinking" 토큰을 생성하며, 이 토큰은 사용자에게 공개되지 않는 것이다 (hidden CoT). 핵심 학습 기법으로는 대규모 RL과 PRM의 결합이 사용된 것으로 추정되나, 구체적 기술 세부사항은 비공개인 것이다.

**벤치마크 결과:**

| 벤치마크 | GPT-4o | o1-preview | o1 |
|----------|--------|------------|-----|
| AIME 2024 (수학 경시대회) | 12.4% | 44.6% | **83.3%** (con@64) |
| Codeforces (프로그래밍) | 11th percentile | 62nd percentile | **89th percentile** |
| GPQA Diamond (대학원 과학) | 53.6% | 73.3% | **78.0%** |
| MATH (수학) | 60.3% | 85.5% | **94.8%** |

con@64는 64개의 샘플을 생성하고 consensus(다수결)로 답변을 선택하는 방식이다.

**핵심 의의 — Test-time Compute Scaling:** o1은 Snell et al. (2024)이 이론적으로 분석한 test-time compute scaling 법칙을 실증한 것이다. 학습 시 연산량(training compute)을 늘리는 대신, 추론 시 연산량(test-time compute)을 늘림으로써 성능을 향상시킬 수 있다는 새로운 scaling 패러다임을 제시한 것이다. 이는 일부 태스크에서 "소형 모델 + 긴 추론 > 대형 모델 + 짧은 추론"이 성립함을 의미하는 것이다.

**새로운 문제:** (1) 추론 지연(latency)이 매우 높다 (수십 초 ~ 수 분). (2) thinking 토큰으로 인한 비용 증가가 상당하다. (3) 비공개 모델이므로 재현 및 학술 연구가 불가능하다. (4) 단순한 문제에도 불필요하게 긴 추론을 수행하는 "overthinking" 문제가 존재하는 것이다.

### 3.14 OpenAI o3

**이전 문제:** o1은 ARC-AGI 벤치마크에서 25%의 성능을 기록하여, 추상적 추론과 일반화 능력에 한계를 노출한 것이다.

**해결:** OpenAI (2025)는 o3를 공개하였으며, 이는 o1 대비 추론 깊이와 정확성을 대폭 향상시킨 것이다. o3의 핵심 특징은 "deliberative alignment" — 모델이 추론 과정에서 자체적으로 안전성 정책(safety policy)을 참조하고 적용하는 능력을 갖추었다는 것이다.

**벤치마크 결과:**
- ARC-AGI: o1 25% → o3 **87.5%** (high-compute setting)
- AIME 2024: o1 83.3% → o3 **96.7%**
- EpochAI Frontier Math: o1 ~9% → o3 **25.2%**
- Codeforces: o1 89th → o3 **175위 수준** (2727 Elo)

**새로운 문제:** high-compute setting에서의 비용이 단일 문제당 수천 달러에 달할 수 있으며, 효율성과 성능 사이의 트레이드오프가 극심한 것이다.

### 3.15 DeepSeek-R1 및 GRPO

**이전 문제:** o1의 학습 방법론이 비공개이므로 오픈소스 커뮤니티에서 Reasoning Model을 재현할 수 없었으며, PPO 기반 RL은 critic model의 학습과 메모리 비용이 과도한 것이다.

**해결:** DeepSeek AI (2025)는 DeepSeek-R1을 공개하여, 오픈소스 Reasoning Model의 첫 성공 사례를 제시한 것이다. R1의 핵심 학습 알고리즘은 GRPO(Group Relative Policy Optimization)이며, 이는 DeepSeekMath (Shao et al., 2024)에서 최초 제안된 것이다.

**GRPO의 기술적 세부사항:**

GRPO는 PPO에서 critic model(value function)을 제거하고, 그룹 내 상대적 보상으로 대체한 알고리즘이다. 구체적 절차는 다음과 같다:

1. 각 질문 qᵢ에 대해 현재 정책 πθ로부터 G개의 응답 {o₁, o₂, ..., o_G}를 샘플링한다.
2. 각 응답에 대해 보상 rᵢ를 계산한다 (규칙 기반 보상: 정답 여부, 형식 준수 등).
3. 그룹 내 보상을 정규화한다: Âᵢ = (rᵢ - mean(r)) / std(r)
4. 정책을 업데이트한다:

```
L_GRPO(θ) = E_q [1/G Σᵢ min(ρᵢÂᵢ, clip(ρᵢ, 1-ε, 1+ε)Âᵢ) - β·D_KL(πθ || πref)]

여기서 ρᵢ = πθ(oᵢ|q) / πθ_old(oᵢ|q)
```

**PPO 대비 GRPO의 이점:**
- Critic model 불필요 → 메모리 사용량 50% 감소
- Reward model 대신 규칙 기반 보상 사용 가능 → 보상 해킹 방지
- 그룹 내 상대 비교 → 보상의 절대값에 민감하지 않음

**DeepSeek-R1-Zero의 발견:**

DeepSeek-R1-Zero는 SFT 없이 순수 RL(GRPO)만으로 학습된 모델이며, 이 실험에서 다음과 같은 주목할 만한 현상이 관찰된 것이다:
- **자발적 CoT 발현:** 모델이 명시적 지시 없이도 단계적 추론 과정을 자율적으로 생성하기 시작한 것이다.
- **"Aha moment":** 학습 과정 중 모델이 "Wait, let me reconsider..."와 같은 자기 성찰(self-reflection) 패턴을 자발적으로 학습한 것이다.
- **추론 길이의 자율 증가:** 학습이 진행됨에 따라 모델의 평균 응답 길이가 자연스럽게 증가한 것이다.

**R1-Zero의 한계와 R1의 해결:**
R1-Zero는 가독성 저하(poor readability), 언어 혼합(language mixing) 문제를 보였으며, 이를 해결하기 위해 DeepSeek-R1은 다단계 학습 파이프라인을 도입한 것이다:
1. Cold start SFT: 소량의 고품질 CoT 데이터로 초기 SFT
2. Reasoning RL: GRPO로 추론 능력 강화
3. Rejection sampling + SFT: RL 체크포인트에서 고품질 추론 데이터 생성 후 재학습
4. General RL: 추론 외 일반 능력(helpfulness, safety)에 대한 RL

**벤치마크 결과:**

| 벤치마크 | DeepSeek-V3 | DeepSeek-R1 | o1 |
|----------|-------------|-------------|-----|
| AIME 2024 | 39.2% | **79.8%** | 79.2% |
| MATH-500 | 90.2% | **97.3%** | 96.4% |
| Codeforces | 58.7th | **96.3rd percentile** | 96.6th |
| GPQA Diamond | 59.1% | **71.5%** | 75.7% |
| GSM8K | 89.3% | **97.0%** | 96.7% (est.) |

**핵심 의의:** (1) 오픈 웨이트(open-weight) 공개로 학술 연구 가속화. (2) RL만으로 추론 능력이 발현됨을 입증하여, 추론 능력의 기원에 대한 이해 심화. (3) GRPO의 효율성으로 중소 연구 그룹에서도 Reasoning Model 학습 가능성을 제시한 것이다.

**새로운 문제:** (1) 학습에 수천 GPU 필요 (여전히 고비용). (2) 비추론 태스크(창작, 대화 등)에서의 성능 저하 가능성. (3) 추론 과정이 항상 사실에 기반하지 않을 수 있음(hallucinated reasoning). (4) R1-Zero에서 관찰된 "자발적 CoT"가 진정한 추론인지 패턴 매칭인지에 대한 학술적 논쟁이 지속되는 것이다.

### 3.16 QwQ (Alibaba)

**이전 문제:** Reasoning Model의 학습에 필요한 인프라와 데이터 요구사항이 매우 높아, 소수의 조직만 접근 가능한 것이다.

**해결:** Alibaba Qwen 팀 (2024)은 QwQ-32B-Preview를 공개한 것이다. QwQ는 Qwen2.5-32B 기반으로 자체 추론 강화학습을 적용한 모델이며, 32B 파라미터로 일부 태스크에서 o1-preview 수준의 성능을 달성한 것이다.

**벤치마크 결과:**
- AIME 2024: 50.0% (o1-preview 44.6% 대비 우수)
- MATH: 90.6%
- LiveCodeBench: 50.0%

**의의:** 중형 모델(32B)에서도 Reasoning Model 구현이 가능함을 입증하였으며, 이는 test-time compute scaling이 모델 크기의 한계를 일정 부분 상쇄할 수 있음을 시사하는 것이다.

### 3.17 Test-time Compute Scaling

**이전 문제:** 기존 scaling law (Kaplan et al., 2020; Hoffmann et al., 2022)는 학습 시 연산량(training FLOPS)과 성능의 관계만 다루었으며, 추론 시 연산량의 역할은 분석되지 않은 것이다.

**해결:** Snell et al. (2024)은 "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters"에서 test-time compute scaling의 체계적 분석을 제시한 것이다. 핵심 발견은 다음과 같다:

1. **Compute-optimal scaling:** 동일한 추론 예산 하에서, 더 작은 모델에 더 많은 test-time compute를 할당하는 것이 더 큰 모델에 적은 compute를 할당하는 것보다 효과적일 수 있다.
2. **두 가지 전략:** (a) Verifier-based (PRM + best-of-N): 여러 답변을 생성하고 PRM으로 최적 답변 선택, (b) Revision-based: 모델이 자체적으로 답변을 수정하도록 반복 유도.
3. **난이도 의존성:** 쉬운 문제에서는 추가 compute의 효용이 빠르게 감소하나, 어려운 문제에서는 compute 증가에 따른 성능 향상이 지속되는 것이다.

### 3.18 Reasoning Verification

**이전 문제:** Reasoning Model이 생성하는 긴 추론 과정의 정확성을 검증할 체계적 방법이 부재한 것이다.

**해결:** 추론 검증 분야에서 다수의 접근법이 제안된 것이다:

1. **PRM 기반 검증:** 각 단계에 대한 자동화된 점수 부여 (Lightman et al., 2024; Wang et al., 2024).
2. **Math-Shepherd:** Pang et al. (2024)은 자동화된 프로세스 보상 레이블링 방법을 제안하여, 인간 레이블링 없이 PRM을 학습 가능하게 한 것이다. 핵심은 각 단계 이후의 completion을 다수 샘플링하여 정답 도달 비율로 단계별 보상을 추정하는 것이다.
3. **Self-verification:** 모델 자체가 생성한 추론 과정을 재검토하도록 유도하는 기법 (Weng et al., 2023).
4. **Formal verification:** 추론 과정을 형식 증명 체계(Lean, Isabelle 등)로 번역하여 기계적으로 검증하는 접근 (Jiang et al., 2023).

**새로운 문제:** 자동 검증의 정확도가 인간 수준에 도달하지 못하며, 형식 검증은 수학/논리에만 적용 가능한 것이다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 제1단계: Prompting 기반 CoT (2022)

```
Direct Prompting → 다단계 추론 실패
  ↓ [문제 인식: 중간 단계 부재]
Few-shot CoT (Wei et al., 2022)
  ↓ [문제: 수동 예시 설계 필요]
Zero-shot CoT (Kojima et al., 2022)
  ↓ [문제: 단일 경로 의존, 성능 한계]
Auto-CoT (Zhang et al., 2023)
```

이 단계의 핵심 기여는 "중간 추론 단계의 명시적 생성이 LLM의 추론 능력을 극적으로 향상시킨다"는 발견이다. 그러나 프롬프팅만으로는 모델의 내재적 능력을 변경하지 못한다는 근본적 한계가 존재하는 것이다.

### 4.2 제2단계: 다중 경로 및 탐색 (2023)

```
Few-shot CoT → 단일 경로 취약성
  ↓ [문제 인식: 하나의 경로가 틀리면 답도 틀림]
Self-Consistency (Wang et al., 2023)
  ↓ [문제: 선형 경로만 가능, 역추적 불가]
Tree of Thoughts (Yao et al., 2023)
  ↓ [문제: 분기 병합 불가]
Graph of Thoughts (Besta et al., 2024)
  ↓ [문제: 탐색 효율성 낮음]
MCTS for Reasoning (Feng et al., 2024; Xie et al., 2024)
```

이 단계의 핵심은 추론 공간의 탐색(search) 개념 도입이다. 그러나 외부 탐색은 연산 비용이 높고, 모델 자체의 추론 능력을 향상시키지는 못한다는 한계가 있는 것이다.

### 4.3 제3단계: 자기 개선 및 학습 기반 (2022-2024)

```
CoT 데이터 부족
  ↓ [문제 인식: 대규모 추론 데이터 수동 구축 불가]
STaR (Zelikman et al., 2022)
  ↓ [문제: QA 태스크에 한정]
Quiet-STaR (Zelikman et al., 2024)
  ↓ [문제: 소형 모델에서 효과 제한]
```

이 단계는 모델의 내재적 추론 능력을 학습 과정에서 향상시키려는 시도이며, 이후 Reasoning Model의 이론적 기반이 된 것이다.

### 4.4 제4단계: 검증 및 보상 모델 (2023-2024)

```
CoT의 불충실성(unfaithfulness)
  ↓ [문제 인식: 추론 과정이 정확한지 검증 불가]
ORM → 최종 답만 평가, 과정 무시
  ↓ [문제: 잘못된 추론으로 정답 도달 가능]
PRM (Lightman et al., 2024)
  ↓ [문제: 인간 레이블링 비용 과다]
Math-Shepherd (Pang et al., 2024) → 자동 PRM 레이블링
```

PRM의 성공은 Reasoning Model에서 보상 신호의 설계가 핵심임을 입증한 것이다.

### 4.5 제5단계: Reasoning Models (2024-2025)

```
프롬프팅/탐색/검증의 한계 수렴
  ↓ [문제 인식: 모델 자체에 추론 능력 내재화 필요]
o1 (OpenAI, 2024) — 비공개 RL 기반 Reasoning Model
  ↓ [문제: 비공개, 재현 불가]
DeepSeek-R1 (2025) — GRPO 기반 오픈소스 Reasoning Model
  ↓ [R1-Zero가 RL만으로 CoT 자발적 발현 입증]
QwQ (Alibaba, 2024) — 32B 중형 모델에서의 Reasoning
o3 (OpenAI, 2025) — 차세대 Reasoning Model, ARC-AGI 87.5%
  ↓ [현재 문제: 효율성, 일반화, 검증]
```

이 단계는 현재 진행 중이며, 핵심 연구 방향은 (1) 추론 효율성 향상(overthinking 방지), (2) 비수학 도메인으로의 일반화, (3) 추론 과정의 충실성 보장인 것이다.

### 4.6 전체 패러다임 전환 요약

| 시기 | 패러다임 | 핵심 질문 |
|------|----------|-----------|
| ~2022 | Prompting | "어떻게 말하면 모델이 잘 추론하는가?" |
| 2023 | Search | "추론 공간을 어떻게 효율적으로 탐색하는가?" |
| 2023-2024 | Verification | "추론 과정을 어떻게 검증하는가?" |
| 2024-2025 | Internalization | "어떻게 모델 자체에 추론 능력을 학습시키는가?" |
| 2025- | Efficiency | "추론 비용을 어떻게 최적화하는가?" |

이 흐름은 "외부 유도 → 외부 탐색 → 외부 검증 → 내재화 → 효율화"라는 일관된 인과 구조를 가지며, 각 단계는 이전 단계의 한계를 해결하면서 새로운 문제를 제기하는 변증법적 구조를 형성하는 것이다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou | **NeurIPS 2022** | https://arxiv.org/abs/2201.11903 |
| 2 | Large Language Models are Zero-Shot Reasoners | Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa | **NeurIPS 2022** | https://arxiv.org/abs/2205.11916 |
| 3 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou | **ICLR 2023** | https://arxiv.org/abs/2203.11171 |
| 4 | Automatic Chain of Thought Prompting in Large Language Models | Zhuosheng Zhang, Aston Zhang, Mu Li, Alex Smola | **ICLR 2023** | https://arxiv.org/abs/2210.03493 |
| 5 | Complexity-Based Prompting for Multi-Step Reasoning | Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, Tushar Khot | **ICLR 2023** | https://arxiv.org/abs/2210.00720 |
| 6 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas Griffiths, Yuan Cao, Karthik Narasimhan | **NeurIPS 2023** | https://arxiv.org/abs/2305.10601 |
| 7 | Graph of Thoughts: Solving Elaborate Problems with Large Language Models | Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Grzeszczak, Tomasz Hoefler | **AAAI 2024** | https://arxiv.org/abs/2308.09687 |
| 8 | Faithful Chain-of-Thought Reasoning | Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, Chris Callison-Burch | **NAACL 2023** | https://arxiv.org/abs/2301.13379 |
| 9 | STaR: Bootstrapping Reasoning With Reasoning | Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah Goodman | **NeurIPS 2022** | https://arxiv.org/abs/2203.14465 |
| 10 | Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking | Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah Goodman | arXiv 2024 | https://arxiv.org/abs/2403.09629 |
| 11 | Let's Verify Step by Step | Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe | **ICLR 2024** | https://arxiv.org/abs/2305.20050 |
| 12 | Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters | Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar | arXiv 2024 | https://arxiv.org/abs/2408.03314 |
| 13 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek AI | arXiv 2025 | https://arxiv.org/abs/2501.12948 |
| 14 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo | arXiv 2024 | https://arxiv.org/abs/2402.03300 |
| 15 | Learning to Reason with LLMs (o1 System Card) | OpenAI | OpenAI Technical Report, 2024 | https://openai.com/index/learning-to-reason-with-llms/ |
| 16 | QwQ: Reflect Deeply on the Boundaries of the Unknown | Qwen Team, Alibaba | Qwen Blog, 2024 | https://qwenlm.github.io/blog/qwq-32b-preview/ |
| 17 | Scaling Laws for Neural Language Models | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei | arXiv 2020 | https://arxiv.org/abs/2001.08361 |
| 18 | Training Verifiers to Solve Math Word Problems | Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman | arXiv 2021 | https://arxiv.org/abs/2110.14168 |
| 19 | Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations | Peiyi Wang, Lei Li, Zhihong Shao, R.X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, Zhifang Sui | **ACL 2024** | https://arxiv.org/abs/2312.08935 |
| 20 | Biases and Unfaithfulness in Chain-of-Thought Reasoning | Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman | arXiv 2024 | https://arxiv.org/abs/2305.04388 |
| 21 | Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective | Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, Liwei Wang | **NeurIPS 2024** | https://arxiv.org/abs/2305.15408 |
| 22 | AlphaZero-Like Tree-Search Can Guide Large Language Model Decoding and Training | Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, Jun Wang | **ICML 2024** | https://arxiv.org/abs/2309.17179 |
| 23 | rStar: Mutual Reinforcement Reasoning in Small Language Models | Zhenting Xie, Qi Sun, Yining Zheng, Hongyu Lin, Xianpei Han, Le Sun | arXiv 2024 | https://arxiv.org/abs/2408.06195 |
| 24 | Training Chain-of-Thought via Latent-Variable Inference | Du Phan, Matthew D. Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Courville, Mohammad Norouzi, Peer Fischer, Dani Yogatama | **NeurIPS 2023** | https://arxiv.org/abs/2312.02179 |
| 25 | Large Language Models Cannot Self-Correct Reasoning Yet | Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou | **ICLR 2024** | https://arxiv.org/abs/2310.01798 |
| 26 | Measuring Mathematical Problem Solving With the MATH Dataset | Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt | **NeurIPS 2021** | https://arxiv.org/abs/2103.03874 |
| 27 | PAL: Program-Aided Language Models | Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig | **ICML 2023** | https://arxiv.org/abs/2211.10435 |
| 28 | Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification | Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Liang Song, Mingjie Zhan, Hongsheng Li | arXiv 2023 | https://arxiv.org/abs/2308.07921 |
| 29 | Chain-of-Thought Reasoning Without Prompting | Xuezhi Wang, Denny Zhou | arXiv 2024 | https://arxiv.org/abs/2402.10200 |
| 30 | V-STaR: Training Verifiers for Self-Taught Reasoners | Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, Rishabh Agarwal | arXiv 2024 | https://arxiv.org/abs/2402.06457 |
| 31 | Reasoning with Language Model is Planning with World Model | Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu | **EMNLP 2023** | https://arxiv.org/abs/2305.14992 |
| 32 | Self-Refine: Iterative Refinement with Self-Feedback | Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark | **NeurIPS 2023** | https://arxiv.org/abs/2303.17651 |
| 33 | Cumulative Reasoning with Large Language Models | Yifan Zhang, Jingqin Yang, Yang Yuan, Andrew Chi-Chih Yao | arXiv 2024 | https://arxiv.org/abs/2308.04371 |
| 34 | Journey Learning: Cooperative Curriculum for Reasoning | Zhihong Shao, Fei Huang, Yongbin Li | **ACL 2024** | https://arxiv.org/abs/2312.09740 |
| 35 | Thinking Fast and Slow with Deep Learning and Tree Search | Thomas Anthony, Zheng Tian, David Barber | **NeurIPS 2017** | https://arxiv.org/abs/1705.08439 |
