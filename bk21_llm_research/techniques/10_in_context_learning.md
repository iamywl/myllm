# In-Context Learning (ICL, 맥락 내 학습)

## 1. 기법의 정의

In-Context Learning(ICL)은 사전 학습된 대규모 언어 모델(LLM)이 gradient update 없이, 추론 시점에 프롬프트 내 제공된 소수의 입출력 예시(demonstration)만으로 새로운 태스크를 수행하는 능력이다. 전통적 기계 학습에서 모델은 학습(training) 단계에서만 파라미터를 갱신하고 추론(inference) 시에는 고정된 함수로 동작하지만, ICL에서는 모델이 추론 시점에 컨텍스트 내 예시로부터 태스크 매핑을 암묵적으로 추론(implicit inference)하는 것이다. 이는 전통적 supervised learning 패러다임과 근본적으로 다른 메타-학습(meta-learning) 형태이며, LLM 시대의 핵심 능력 중 하나로 간주된다.

형식적으로, ICL은 입력 $x$에 대한 예측을 수행할 때 $k$개의 demonstration $(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k)$를 프롬프트에 연결(concatenation)하여 조건부 확률 $P(y|x_1, y_1, \ldots, x_k, y_k, x)$를 모델링하는 것이다. 이때 모델 파라미터 $\theta$는 고정되어 있으며, 오직 입력 컨텍스트만 변경된다.

---

## 2. 기존 기법의 한계와 ICL 등장 배경

### 2.1 Fine-tuning 패러다임의 구조적 문제

**문제:** 사전 학습(pre-training) 후 각 downstream 태스크에 대해 별도의 fine-tuning을 수행하는 BERT 시대(Devlin et al., 2019)의 패러다임은 다음과 같은 구조적 한계를 가지고 있었다.

- **태스크별 학습 비용**: 새로운 태스크마다 labeled 데이터 수집, GPU 학습, 하이퍼파라미터 탐색이 필요하다. GPT-3 175B 규모 모델의 full fine-tuning은 수백 GPU-hour가 소요된다.
- **모델 저장 비용**: $N$개 태스크에 대해 $N$개의 fine-tuned 모델 체크포인트를 유지해야 하며, 175B 모델 기준 체크포인트 하나당 약 350GB이다.
- **도메인 전환 지연**: 새로운 태스크 등장 시 데이터 수집부터 배포까지 수일~수주가 소요되어 실시간 대응이 불가능하다.
- **catastrophic forgetting**: 특정 태스크에 fine-tuning하면 일반 능력이 저하되는 현상이 발생한다.

**해결 시도:** Adapter(Houlsby et al., 2019), LoRA(Hu et al., 2022) 등 parameter-efficient fine-tuning(PEFT) 기법이 제안되었으나, 여전히 태스크별 학습 과정 자체가 필요하다는 근본적 한계는 해소되지 않았다.

### 2.2 GPT-3에 의한 ICL의 발견

**해결:** Brown et al. (2020)은 GPT-3 논문에서 모델 규모를 175B 파라미터까지 확대했을 때, 프롬프트에 몇 개의 예시를 제공하는 것만으로 gradient update 없이 다양한 NLP 태스크를 수행할 수 있음을 발견하였다. 이것이 ICL의 공식적 출발점이다.

GPT-3의 핵심 실험 결과는 다음과 같다:
- **Few-shot (32 예시)** 설정에서 SuperGLUE 벤치마크 71.8점을 달성하였으며, 이는 fine-tuned BERT-Large(69.0점)를 초과하는 수치이다.
- **Zero-shot** 설정에서도 다수 태스크에서 SOTA fine-tuned 모델의 80% 이상 성능을 달성하였다.
- 번역, 산술, 단어 재배열 등 사전에 명시적으로 학습하지 않은 태스크에서도 동작하였다.

**새로운 문제:** ICL의 존재가 확인되었으나, (1) 왜 동작하는지 이론적 설명이 부재하고, (2) 예시 선택/순서에 따른 성능 변동이 극심하며, (3) 소형 모델에서는 동작하지 않는다는 문제가 발생하였다.

### 2.3 창발적 능력으로서의 ICL

**문제:** ICL이 모든 규모의 모델에서 동작하는 것이 아니라는 관찰이 보고되었다.

Wei et al. (2022b)의 emergent abilities 연구에 따르면, ICL 능력은 모델 규모가 특정 임계점을 넘어야 출현하는 창발적(emergent) 속성이다. 구체적으로:
- **10B 이하** 모델에서는 few-shot 예시 추가 시 성능 향상이 미미하거나 없다.
- **62B 이상**(PaLM 기준)에서 ICL 성능이 급격히 향상되는 phase transition이 관찰된다.
- 이 현상은 단순한 scaling law의 연장이 아니라, 질적으로 새로운 능력의 출현이다.

**새로운 문제:** 창발적 능력의 존재는 확인되었으나, 이것이 진정한 불연속적 phase transition인지, 아니면 평가 metric의 비선형성에 의한 착시인지에 대한 논쟁이 발생하였다. Schaeffer et al. (2023)은 metric 선택에 따라 emergent ability가 사라질 수 있음을 보였으며, 이 논쟁은 현재까지 지속 중이다.

---

## 3. 주요 기법 상세

### 3.1 ICL의 기본 형태와 분류

```
Few-shot ICL 예시:
  입력: "행복하다" → 출력: "긍정"
  입력: "슬프다"   → 출력: "부정"
  입력: "기쁘다"   → 출력: ?   ← 모델이 패턴을 추론하여 "긍정" 출력
```

| 유형 | 예시 수 | 특징 | 대표 성능 (SST-2, GPT-3 175B) |
|------|---------|------|-------------------------------|
| **Zero-shot** | 0 | 태스크 설명(instruction)만 제공 | 81.4% |
| **One-shot** | 1 | 단일 demonstration | 90.1% |
| **Few-shot** | 2~64 | 소수 demonstration | 95.3% (32-shot) |
| **Many-shot** | 수백~수천 | 긴 컨텍스트 윈도우 활용 | fine-tuning 수준 |

### 3.2 예시 선택 전략 (Example Selection)

**문제:** 동일한 모델과 동일한 예시 수에서도, 어떤 예시를 선택하느냐에 따라 성능이 최대 20% 이상 변동하는 현상이 관찰되었다 (Liu et al., 2022).

#### 3.2.1 유사도 기반 선택

**해결:** Liu et al. (2022)은 KATE(kNN-Augmented in-conText Example selection)를 제안하였다. 테스트 입력과 의미적으로 유사한 예시를 sentence embedding 공간에서 kNN으로 검색하여 선택하는 방법이다.

- GPT-3 기준 SST-2에서 random 선택 대비 +5.2% 정확도 향상을 달성하였다.
- 유사도 계산에 Sentence-BERT 임베딩을 사용하였다.

**새로운 문제:** 유사도 기반 선택은 테스트 입력의 분포가 학습 예시와 유사할 때만 효과적이며, 반례(counter-example)나 다양성(diversity)이 필요한 태스크에서는 오히려 성능이 저하될 수 있다.

#### 3.2.2 다양성 기반 선택

**해결:** Levy et al. (2023)은 예시의 다양성을 고려하는 선택 전략이 일부 태스크에서 유사도 기반보다 우수함을 보였다. 입력 공간의 커버리지를 극대화하는 방식으로, 클러스터링 기반 대표 예시 선택이 이에 해당한다.

#### 3.2.3 학습 기반 선택

**해결:** Rubin et al. (2022)은 EPR(Efficient Prompt Retrieval)을 제안하여, 예시 선택 자체를 학습 가능한 retriever로 수행하였다. Contrastive learning으로 학습된 retriever가 주어진 테스트 입력에 대해 ICL 성능을 극대화하는 예시를 검색한다.

- GPT-Neo 2.7B에서 random 선택 대비 다수 태스크에서 +8~15% 성능 향상이 관찰되었다.

**새로운 문제:** 학습 기반 retriever는 그 자체로 태스크별 학습이 필요하다는 점에서, ICL의 "학습 불필요" 장점을 일부 상쇄한다.

### 3.3 예시 순서 민감성 (Example Ordering Sensitivity)

**문제:** Lu et al. (2022)은 동일한 예시 집합이라도 순서에 따라 성능이 최대 30%까지 변동함을 발견하였다. GPT-3 175B에서 SST-2 few-shot 정확도가 예시 순서에 따라 54.3%~93.4% 범위를 보였다.

**원인 분석:**
- **Recency bias**: 모델이 프롬프트 끝부분의 예시에 과도한 가중치를 부여하는 경향이 있다. 마지막 예시의 레이블이 다음 예측에 불균형적 영향을 미친다.
- **Majority label bias**: 특정 레이블이 예시에서 다수를 차지하면 해당 레이블로 편향된다.

**해결:** Lu et al. (2022)은 GlobalE 및 LocalE 메트릭을 제안하여 엔트로피 기반으로 최적 순서를 탐색하는 방법을 제시하였다. Calibration 기법(Zhao et al., 2021)도 이 문제를 완화하는 데 효과적이다. Zhao et al.은 content-free 입력("N/A")에 대한 모델 출력의 편향을 측정하고 이를 보정하는 contextual calibration을 제안하여, GPT-3에서 평균 8%p의 성능 향상을 달성하였다.

**새로운 문제:** 최적 순서 탐색은 가능한 순열(permutation) 수가 $k!$로 폭발적이므로, 예시 수가 증가하면 실용적이지 않다.

### 3.4 Min et al.의 핵심 발견: 랜덤 레이블에서도 ICL이 동작

**문제:** ICL이 예시의 입출력 매핑(input-label mapping)을 학습하는 것인지, 아니면 다른 무언가를 활용하는 것인지가 불명확하였다.

**핵심 실험 (Min et al., 2022):** 예시의 레이블을 완전히 랜덤하게 치환해도 ICL 성능이 크게 저하되지 않는다는 실험 결과를 보고하였다.

구체적 수치(GPT-3 davinci 기준):
- **정상 레이블**: SST-2 94.0%, Subj 91.4%, AGNews 76.8%
- **랜덤 레이블**: SST-2 89.1%, Subj 84.0%, AGNews 67.5%
- 성능 저하는 존재하나, random baseline(50%, 50%, 25%) 대비 매우 높은 수치를 유지한다.

이 실험으로부터 ICL에서 중요한 요소의 우선순위가 도출되었다:
1. **입력-레이블 공간(input-label space)의 분포 정보** — 가장 중요
2. **입력 텍스트의 형식(format)과 도메인** — 매우 중요
3. **실제 입출력 매핑(ground-truth mapping)** — 기여하지만 필수적이지 않음

**새로운 문제:** 이 발견은 ICL의 메커니즘에 대한 기존 직관("예시의 정답으로부터 학습")을 근본적으로 도전하며, ICL이 실제로 무엇을 하는 것인지에 대한 이론적 설명의 필요성을 부각시켰다.

### 3.5 ICL의 이론적 설명

Min et al.의 발견 이후, ICL의 동작 메커니즘을 설명하려는 다수의 이론적 프레임워크가 제안되었다. 이들은 상호 배타적이라기보다 상보적 관점을 제공한다.

#### 3.5.1 Implicit Gradient Descent 가설

**핵심 주장:** Transformer의 forward pass가 내부적으로 gradient descent를 수행하고 있다는 가설이다.

Von Oswald et al. (2023)은 선형 회귀 문제에서 Transformer가 ICL을 수행할 때, 내부 연산이 gradient descent의 단계와 수학적으로 동치임을 증명하였다. 구체적으로, 선형 회귀 손실 $\mathcal{L}(W) = \frac{1}{2k}\sum_{i=1}^{k} \|Wx_i - y_i\|^2$에 대해 단일 self-attention layer의 연산이 다음과 동등하다:

$$W_{t+1} = W_t - \eta \nabla_W \mathcal{L}(W_t; \{(x_i, y_i)\}_{i=1}^k) = W_t - \frac{\eta}{k} \sum_{i=1}^{k}(W_t x_i - y_i)x_i^\top$$

여기서 attention 가중치가 학습률 $\eta$의 역할을 수행하며, demonstration이 gradient 계산의 데이터 역할을 한다. 다층 Transformer의 경우 각 layer가 하나의 gradient step에 대응하여, $L$-layer Transformer가 $L$-step gradient descent를 구현한다.

Dai et al. (2023)은 이를 실제 LLM 설정으로 확장하여, ICL과 explicit fine-tuning 간의 dual form 관계를 증명하였다. Transformer attention의 연산을 분석하면, ICL은 demonstration에 대한 meta-gradient를 계산하고 이를 attention 가중치에 적용하는 것과 동등하다:

$$\text{ICL: } \text{Attn}(q, \{(x_i, y_i)\}) \approx W_0 q + \Delta W \cdot q, \quad \Delta W = \sum_{i=1}^{k} \nabla_{W} \mathcal{L}(W; x_i, y_i) \bigg|_{W=W_0}$$

Akyürek et al. (2023)은 이를 확장하여, Transformer가 ICL 시 mesa-optimizer를 구현할 수 있음을 보였다. 선형 모델에서 Transformer가 ordinary least squares ($\hat{W} = (X^\top X)^{-1}X^\top Y$), ridge regression ($\hat{W} = (X^\top X + \lambda I)^{-1}X^\top Y$), 심지어 gradient descent with momentum까지 구현 가능함을 실험적으로 확인하였다.

**한계:** 이 분석은 선형 모델 및 합성(synthetic) 데이터 설정에 제한되어 있으며, 실제 LLM의 자연어 ICL에 직접 적용할 수 있는지는 미해결 문제이다. 비선형 태스크에서의 ICL 동작이 gradient descent와 동치인지에 대한 이론적 증명은 아직 존재하지 않는다.

#### 3.5.2 Bayesian Inference 가설

**핵심 주장:** ICL은 사전 학습 데이터의 분포에 대한 암묵적 Bayesian 추론 과정이다.

Xie et al. (2022)은 ICL을 다음과 같이 형식화하였다: 사전 학습 데이터가 잠재 개념(latent concept) $\theta$에 의해 생성되는 Hidden Markov Model로 모델링될 때, ICL은 demonstration을 관찰하여 사후 분포를 업데이트하는 Bayesian 추론 과정이다. 구체적으로, 사전 학습 분포 $p_{\text{pretrain}}$에서 잠재 개념 $\theta$에 대한 사전 분포 $p(\theta)$가 유도되며, $k$개의 demonstration을 관찰한 후의 예측은 다음과 같이 표현된다:

$$P(y|x_1, y_1, \ldots, x_k, y_k, x) = \int_\Theta P(y|x, \theta) \cdot P(\theta|x_1, y_1, \ldots, x_k, y_k) \, d\theta$$

여기서 사후 분포는 Bayes' rule에 의해 다음과 같이 업데이트된다:

$$P(\theta|x_1, y_1, \ldots, x_k, y_k) \propto P(\theta) \prod_{i=1}^{k} P(x_i, y_i|\theta)$$

이 프레임워크는 Min et al.의 랜덤 레이블 실험을 설명할 수 있다: 입력 텍스트 $x_i$ 자체가 잠재 개념 $\theta$에 대한 강한 evidence를 제공하므로, $P(\theta|x_1, \ldots, x_k) \approx P(\theta|x_1, y_1, \ldots, x_k, y_k)$가 성립하여 레이블이 랜덤이어도 올바른 개념을 locate 할 수 있다. 즉, 입력의 분포 정보만으로도 잠재 개념에 대한 사후 확률이 충분히 집중(concentrate)된다.

**한계:** 실제 LLM의 사전 학습 데이터가 HMM 가정에 부합하는지 검증이 어렵다. 또한 실제 자연어의 잠재 개념 공간 $\Theta$가 이산적인지 연속적인지, 그 차원이 어느 정도인지에 대한 실증적 분석이 부족하다.

#### 3.5.3 Task Location 가설

**핵심 주장:** ICL은 새로운 태스크를 "학습"하는 것이 아니라, 사전 학습 시 이미 습득한 다수의 태스크 중에서 올바른 태스크를 "위치 지정(locate)"하는 것이다.

Pan et al. (2023)은 이 가설을 실험적으로 검증하여, 사전 학습 데이터에 포함된 태스크에 대해서는 ICL이 효과적으로 동작하지만, 사전 학습 분포에서 벗어난 태스크에 대해서는 ICL 성능이 급격히 저하됨을 보였다. 이는 ICL이 "학습"보다는 "검색(retrieval)"에 가깝다는 해석을 지지한다.

#### 3.5.4 Induction Heads 메커니즘

**핵심 주장:** ICL 능력은 Transformer 내 특정 attention head 패턴인 induction heads에 의해 구현된다.

Olsson et al. (2022)은 Anthropic에서 수행한 대규모 mechanistic interpretability 연구에서, 2-layer attention 회로가 ICL의 핵심 메커니즘임을 발견하였다:

1. **이전 토큰 헤드(previous token head)**: 현재 토큰의 이전 위치를 참조한다.
2. **인덕션 헤드(induction head)**: "[A][B]...[A] → [B]" 패턴을 완성한다. 즉, 이전에 A 다음에 B가 나온 패턴을 발견하면, 현재 A 다음에 B를 예측한다.

핵심 실험 결과:
- 학습 과정에서 induction heads가 형성되는 시점과 ICL 능력이 출현하는 시점이 정확히 일치한다 (phase transition).
- Induction heads를 ablation(제거)하면 ICL 성능이 급격히 저하된다.
- 이 현상은 1-layer 모델에서는 관찰되지 않으며, 최소 2-layer 이상의 구성적(compositional) 구조가 필요하다.

**새로운 문제:** Induction heads는 패턴 복사(pattern copying) 수준의 ICL을 설명하지만, 추상적 추론이나 태스크 일반화를 요구하는 복잡한 ICL을 완전히 설명하지는 못한다.

### 3.6 Many-shot ICL

**문제:** 기존 ICL 연구는 컨텍스트 윈도우 제한(GPT-3: 2K~4K 토큰)으로 인해 수십 개 이하의 예시만 제공할 수 있었다. 이로 인해 ICL의 성능 상한이 fine-tuning에 미치지 못하는 것으로 간주되었다.

**해결:** Agarwal et al. (2024, Google DeepMind)은 Gemini 1.5 Pro(1M 토큰 컨텍스트)를 활용한 Many-shot ICL을 제안하였다. 수백~수천 개의 demonstration을 제공하여 ICL의 성능 한계를 극복하는 접근이다.

핵심 실험 결과:
- 예시 수 증가에 따라 **log-linear** 성능 향상이 관찰된다. 즉, 예시 수를 10배로 늘릴 때 일정한 성능 향상이 달성된다.
- **MATH 벤치마크**: few-shot(4예시) 45.2% → many-shot(512예시) 61.8% (+16.6%p)
- **GSM8K**: few-shot 78.3% → many-shot(256예시) 90.1% (+11.8%p)
- 일부 태스크에서 **fine-tuning 수준의 성능**을 달성하였다. 특히 분류(classification) 태스크에서 이 경향이 두드러진다.
- 예시 수가 수천을 초과하면 성능 향상이 포화(saturation)되는 현상도 관찰된다.

**새로운 문제:** (1) 수천 개 예시를 포함한 프롬프트의 추론 비용이 매우 높다(토큰 수 × 단가). (2) 긴 컨텍스트에서의 "lost in the middle" 현상(Liu et al., 2024)으로 중간 위치의 예시가 무시될 수 있다. (3) 수천 개의 고품질 labeled 예시가 필요하므로 데이터 효율성 장점이 감소한다.

### 3.7 Reinforced ICL

**문제:** Many-shot ICL은 대량의 human-labeled 예시가 필요하며, 이는 데이터 수집 비용을 증가시킨다.

**해결:** Agarwal et al. (2024)은 Many-shot ICL 논문 내에서 Reinforced ICL을 함께 제안하였다. 핵심 아이디어는 모델 자체가 생성한 Chain-of-Thought(CoT) 추론을 demonstration으로 활용하는 자기강화(self-reinforcement) 방식이다.

절차:
1. 소수의 seed 예시로 모델이 다수의 문제에 대해 CoT 풀이를 생성한다.
2. 정답과 일치하는 풀이만 필터링한다.
3. 필터링된 (문제, 모델 생성 CoT, 정답) 쌍을 demonstration으로 사용한다.

실험 결과:
- Human-labeled rationale 없이도 Many-shot ICL과 유사한 성능을 달성하였다.
- MATH 벤치마크에서 human rationale 대비 -2.1%p 이내의 성능 차이를 보였다.

**새로운 문제:** 모델 자체의 오류가 demonstration에 포함될 수 있으며, 이로 인한 error propagation이 발생할 수 있다. 또한 모델이 이미 잘 풀 수 있는 문제의 rationale만 생성되므로, 어려운 문제에 대한 개선 효과가 제한적이다.

### 3.8 ICL vs Fine-tuning: 체계적 비교

**문제:** ICL과 fine-tuning 중 어떤 접근이 주어진 상황에서 최적인지에 대한 체계적 비교가 부족하였다.

Mosbach et al. (2023)과 다수의 비교 연구가 수행되었으며, 핵심 trade-off는 다음과 같다:

| 차원 | ICL | Fine-tuning |
|------|-----|-------------|
| **학습 비용** | 0 (gradient update 없음) | 수 시간~수일 GPU 학습 |
| **추론 비용** | 높음 (예시가 토큰 소모) | 낮음 (예시 불필요) |
| **데이터 요구량** | 4~수천 예시 | 수백~수만 예시 |
| **성능 상한** | fine-tuning 대비 낮음 (일반적) | 충분한 데이터 시 더 높음 |
| **태스크 전환** | 즉시 (프롬프트만 변경) | 모델 교체 필요 |
| **도메인 적응** | 제한적 (사전 학습 분포 내) | 강함 (새 분포 학습 가능) |
| **모델 크기 의존** | 대형 모델 필수 (>10B) | 소형 모델에서도 가능 |
| **분포 외 일반화** | 사전 학습 분포 내 태스크에 제한 | 새 분포에 적응 가능 |

**실용적 의사결정 기준:**
- 데이터 < 100개, 빠른 프로토타이핑이 필요한 경우 → ICL
- 데이터 > 1000개, 반복적 추론이 필요한 경우 → Fine-tuning (추론 비용 절감)
- 다수 태스크를 동시 지원해야 하는 경우 → ICL (단일 모델)
- 도메인 특화 용어/지식이 필요한 경우 → Fine-tuning

### 3.9 ICL in Code Generation

**문제:** 코드 생성 태스크에서 ICL의 효과는 자연어 태스크와 다른 패턴을 보이며, 별도의 분석이 필요하다. 자연어 ICL에서는 입출력 예시의 의미적 유사성이 핵심이지만, 코드는 구조적 의존성(import, 타입 정의, API 호출 패턴)이 성능에 더 큰 영향을 미치므로, demonstration 선택 전략이 근본적으로 달라야 한다.

**해결:** Nashid et al. (2023)은 코드 관련 few-shot learning에서 retrieval 기반 예시 선택 전략(CEDAR)을 제안하였다. 임베딩 유사도 및 BM25 빈도 분석을 활용하여 테스트 입력과 관련성이 높은 코드 demonstration을 자동으로 검색하는 방법이다.

핵심 실험 결과:
- HumanEval 벤치마크에서 관련 코드 예시를 ICL로 제공하면 pass@1이 +12.2%p 향상되었다 (Codex 기준).
- BM25 기반 코드 검색으로 선택한 예시가 random 선택 대비 +8.7%p 우수하였다.
- 함수 시그니처와 docstring이 핵심 demonstration 역할을 수행하며, 코드의 구조적 특성상 입출력 예시보다 API 패턴과 코딩 스타일이 더 중요하다.

**새로운 문제:** 단일 파일 수준의 예시 선택은 효과적이나, 실제 소프트웨어 개발에서는 다수 파일에 걸친 cross-file 의존성이 존재하므로, repository 전체의 맥락을 반영하는 ICL이 필요하다.

**해결:** Zhang et al. (2023)은 RepoCoder를 제안하여, repository 수준의 코드 완성을 iterative retrieval-generation 파이프라인으로 해결하였다. 동일 repository 내 다른 파일의 코드를 반복적으로 검색하여 컨텍스트로 제공함으로써, 코딩 컨벤션과 프로젝트 구조를 반영한 코드를 생성한다.

핵심 실험 결과:
- Repository 내 cross-file context를 제공하면 코드 완성 정확도가 In-File baseline 대비 10% 이상 향상된다.
- 이는 ICL이 단순 태스크 수행을 넘어, 프로젝트 수준의 맥락 이해에도 활용될 수 있음을 시사한다.

**새로운 문제:** 코드의 긴 의존성(long-range dependency) — 예를 들어 수백 줄 떨어진 타입 정의 참조 — 을 ICL로 처리하기 어려우며, 컨텍스트 윈도우의 효율적 활용이 과제이다. 또한 repository 규모가 커질수록 관련 코드 검색의 정밀도가 저하되며, 어떤 cross-file context가 현재 코드 완성에 실질적으로 기여하는지 판별하는 것이 어렵다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 ICL 발견과 초기 탐색 (2020~2021)

```
Fine-tuning 패러다임의 비효율 (태스크별 학습 필요)
  → GPT-3 (Brown et al., 2020): 175B 모델에서 ICL 능력 발견
    → Zero/One/Few-shot 설정의 체계적 벤치마크
    → 문제: 왜 동작하는지 설명 불가, 예시 민감성 높음
```

### 4.2 ICL 동작 조건 분석 (2021~2022)

```
ICL 성능의 높은 변동성 문제
  → Liu et al. (2022): 유사도 기반 예시 선택으로 안정화
  → Lu et al. (2022): 순서 민감성 발견 (30% 변동), 엔트로피 기반 최적 순서 탐색
  → Zhao et al. (2021): contextual calibration으로 편향 보정
    → 문제: 여전히 ICL의 내부 메커니즘이 불명확
```

### 4.3 ICL 메커니즘 규명 (2022~2023)

```
Min et al. (2022): 랜덤 레이블에서도 ICL 동작 → 입출력 매핑이 핵심이 아님
  → Xie et al. (2022): Bayesian inference로 해석 (잠재 개념의 사후확률 추론)
  → Von Oswald et al. (2023): implicit gradient descent로 해석 (선형 모델 설정)
  → Akyürek et al. (2023): mesa-optimizer 관점에서의 분석
  → Olsson et al. (2022): induction heads의 mechanistic 발견
  → Dai et al. (2023): dual form으로 ICL ≈ implicit fine-tuning 증명
    → 문제: 이론적 분석이 단순화된 설정에 제한, 실제 LLM 적용성 미검증
```

### 4.4 ICL 확장과 실용화 (2023~2025)

```
컨텍스트 윈도우 확대 (Gemini 1M, Claude 200K, GPT-4 128K)
  → Agarwal et al. (2024): Many-shot ICL — 수백~수천 예시로 fine-tuning 수준 달성
  → Reinforced ICL: 모델 자체 생성 rationale로 self-improvement
  → Li et al. (2023): Unified ICL — 다양한 모달리티로 ICL 확장
    → 문제: 추론 비용 증가, lost-in-the-middle 현상
      → 연구 방향: 효율적 컨텍스트 압축, selective attention
```

### 4.5 ICL의 현재 연구 프론티어 (2024~)

현재 ICL 연구는 다음 방향으로 진행 중이다:

1. **Structured ICL**: 단순 입출력 쌍이 아닌 구조화된 demonstration(테이블, 그래프, 코드)을 활용한 ICL이 탐색되고 있다.
2. **ICL과 Fine-tuning의 융합**: ICL로 프로토타이핑 후 fine-tuning으로 성능을 고정하는 하이브리드 파이프라인이 실용화되고 있다.
3. **Multimodal ICL**: 이미지-텍스트 쌍을 demonstration으로 제공하는 비전-언어 ICL이 GPT-4V, Gemini 등에서 활발히 연구되고 있다 (Alayrac et al., 2022).
4. **ICL 안전성**: Adversarial demonstration을 통한 ICL 공격(Wei et al., 2023)과 이에 대한 방어 연구가 진행 중이다.
5. **Batch ICL**: 다수의 테스트 입력을 동시에 프롬프트에 포함하여 처리하는 효율적 ICL 방식이 제안되고 있다 (Lin et al., 2024).
6. **ICL State/Function Vector**: Hendel et al. (2023)의 task vector 연구를 확장하여, ICL의 기능을 단일 벡터로 압축(compress)하는 연구가 활발하다. Liu et al. (2024b)은 inner optimization과 momentum을 적용한 state vector 집계 방법을 제안하여, 수백 개의 demonstration을 단일 벡터로 압축하면서도 many-shot ICL과 유사한 성능을 달성하였다.
7. **Multimodal Task Vector**: Jiang et al. (2024b)은 multimodal ICL에서 task vector를 활용하여 이미지-텍스트 demonstration을 attention head의 compact representation으로 압축하는 Multimodal Task Vector(MTV)를 제안하였다. 이를 통해 컨텍스트 길이 제한을 극복하고 many-shot multimodal ICL을 가능하게 하였다.
8. **ICL의 층별 압축-표현 메커니즘**: 최근 연구에서 ICL 과정이 Transformer의 하위 층에서 태스크 정보를 압축(compression)하고, 상위 층에서 이를 기반으로 출력을 표현(expression)하는 이중 단계 구조를 따른다는 발견이 보고되고 있다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 1 | **Language Models are Few-Shot Learners** | Tom Brown, Benjamin Mann, Nick Ryder et al. (OpenAI) | **NeurIPS 2020** | https://arxiv.org/abs/2005.14165 |
| 2 | **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** | Sewon Min, Xinxi Lyu, Ari Holtzman et al. (UW/Meta) | **EMNLP 2022** | https://arxiv.org/abs/2202.12837 |
| 3 | **What learning algorithm is in-context learning? Investigations with linear models** | Ekin Akyürek, Dale Schuurmans, Jacob Andreas et al. (MIT) | **ICLR 2023** | https://arxiv.org/abs/2211.15661 |
| 4 | **An Explanation of In-context Learning as Implicit Bayesian Inference** | Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma (Stanford) | **NeurIPS 2022** | https://arxiv.org/abs/2111.02080 |
| 5 | **Many-Shot In-Context Learning** | Rishabh Agarwal, Avi Singh, Lei M. Zhang et al. (Google DeepMind) | **ICML 2024** | https://arxiv.org/abs/2404.11018 |
| 6 | **Transformers as Algorithms: Generalization and Stability in In-context Learning** | Yingcong Li, M. Emrullah Ildiz, Dimitris Papailiopoulos, Samet Oymak | **ICML 2023** | https://arxiv.org/abs/2301.07067 |
| 7 | **A Survey on In-context Learning** | Qingxiu Dong, Lei Li, Damai Dai et al. (Chinese Academy of Sciences) | arXiv 2023 | https://arxiv.org/abs/2301.00234 |
| 8 | **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity** | Yao Lu, Max Bartolo, Alastair Moore et al. | **ACL 2022** | https://arxiv.org/abs/2104.08786 |
| 9 | **In-context Reinforcement Learning with Algorithm Distillation** | Michael Laskin, Luyu Wang, Junhyuk Oh et al. (DeepMind) | **ICLR 2023** | https://arxiv.org/abs/2210.14215 |
| 10 | **Transformers Learn In-Context by Gradient Descent** | Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo et al. (Google DeepMind) | **ICML 2023** | https://arxiv.org/abs/2212.07677 |
| 11 | **In-context Learning and Induction Heads** | Catherine Olsson, Nelson Elhage, Neel Nanda et al. (Anthropic) | **Transformer Circuits Thread, 2022** | https://arxiv.org/abs/2209.11895 |
| 12 | **Making Pre-trained Language Models Better Few-shot Learners** | Tianyu Gao, Adam Fisch, Danqi Chen (Princeton) | **ACL 2021** | https://arxiv.org/abs/2012.15723 |
| 13 | **Calibrate Before Use: Improving Few-Shot Performance of Language Models** | Tony Z. Zhao, Eric Wallace, Shibani Santurkar et al. (Stanford) | **ICML 2021** | https://arxiv.org/abs/2102.09690 |
| 14 | **Learning To Retrieve Prompts for In-Context Learning** | Ohad Rubin, Jonathan Herzig, Jonathan Berant (Tel Aviv Univ.) | **NAACL 2022** | https://arxiv.org/abs/2112.08633 |
| 15 | **What Makes Good In-Context Examples for GPT-3?** | Jiachang Liu, Dinghan Shen, Yizhe Zhang et al. (Microsoft) | **DeeLIO Workshop, ACL 2022** | https://arxiv.org/abs/2101.06804 |
| 16 | **Emergent Abilities of Large Language Models** | Jason Wei, Yi Tay, Rishi Bommasani et al. (Google) | **TMLR 2022** | https://arxiv.org/abs/2206.07682 |
| 17 | **Are Emergent Abilities of Large Language Models a Mirage?** | Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford) | **NeurIPS 2023** | https://arxiv.org/abs/2304.15004 |
| 18 | **Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection** | Yu Bai, Fan Chen, Huan Wang et al. (Salesforce) | **NeurIPS 2023** | https://arxiv.org/abs/2306.04637 |
| 19 | **Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers** | Damai Dai, Yutao Sun, Li Dong et al. (Microsoft/CAS) | **ACL 2023 Findings** | https://arxiv.org/abs/2212.10559 |
| 20 | **Larger language models do in-context learning differently** | Jerry Wei, Jason Wei, Yi Tay et al. (Google) | arXiv 2023 | https://arxiv.org/abs/2303.03846 |
| 21 | **Lost in the Middle: How Language Models Use Long Contexts** | Nelson F. Liu, Kevin Lin, John Hewitt et al. (Stanford) | **TACL 2024** | https://arxiv.org/abs/2307.03172 |
| 22 | **Flamingo: a Visual Language Model for Few-Shot Learning** | Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc et al. (DeepMind) | **NeurIPS 2022** | https://arxiv.org/abs/2204.14198 |
| 23 | **Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning** | Noor Nashid, Mifta Sintaha, Ali Mesbah (UBC) | **ICSE 2023** | https://arxiv.org/abs/2305.07496 |
| 24 | **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation** | Fengji Zhang, Bei Chen, Yue Zhang et al. (Microsoft) | **EMNLP 2023** | https://arxiv.org/abs/2303.12570 |
| 25 | **In-context Learning with Many Demonstration Examples** | Mukai Li, Shansan Gong, Jiangtao Feng et al. | arXiv 2023 | https://arxiv.org/abs/2302.04931 |
| 26 | **Diverse Demonstrations Improve In-context Compositional Generalization** | Itay Levy, Ben Bogin, Jonathan Berant (Tel Aviv Univ.) | **ACL 2023** | https://arxiv.org/abs/2212.06800 |
| 27 | **The Learnability of In-Context Learning** | Noam Wies, Yoav Levine, Amnon Shashua (Hebrew Univ./AI21 Labs) | **NeurIPS 2023** | https://arxiv.org/abs/2303.07895 |
| 28 | **Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning** | Lean Wang, Lei Li, Damai Dai et al. (CAS) | **EMNLP 2023** | https://arxiv.org/abs/2305.14160 |
| 29 | **Batch Prompting: Efficient Inference with Large Language Model APIs** | Zhoujian Lin, Haotian Zhang, Andrea Madotto et al. | **EMNLP 2023** | https://arxiv.org/abs/2301.08721 |
| 30 | **In-Context Learning Creates Task Vectors** | Roee Hendel, Mor Geva, Amir Globerson (Tel Aviv Univ.) | **EMNLP 2023 Findings** | https://arxiv.org/abs/2310.15916 |
| 31 | **Scaling Data-Constrained Language Models** | Niklas Muennighoff, Alexander Rush, Boaz Barak et al. (Hugging Face) | **NeurIPS 2023** | https://arxiv.org/abs/2305.16264 |
| 32 | **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google) | **NAACL 2019** | https://arxiv.org/abs/1810.04805 |
| 33 | **PaLM: Scaling Language Modeling with Pathways** | Aakanksha Chowdhery, Sharan Narang, Jacob Devlin et al. (Google) | **JMLR 2023** | https://arxiv.org/abs/2204.02311 |
| 34 | **CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion** | Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad et al. (Columbia/Microsoft) | **NeurIPS 2023 D&B** | https://arxiv.org/abs/2310.11248 |
| 35 | **How does in-context learning work? A framework for understanding the differences from traditional learning** | Sang Michael Xie, Sewon Min (Stanford/UW) | **ICLR 2022 Workshop** | - |
| 36 | **In-Context Learning State Vector with Inner and Momentum Optimization** | Dongfang Liu, Yichen Wen, Junyan Li et al. | **NeurIPS 2024** | https://arxiv.org/abs/2404.11225 |
| 37 | **Multimodal Task Vectors Enable Many-Shot Multimodal In-Context Learning** | Brandon Huang, Chancharik Mitra, Assaf Arbelle et al. | **NeurIPS 2024** | https://arxiv.org/abs/2406.15334 |
| 38 | **In-Context Autoencoder for Context Compression in a Large Language Model** | Tao Ge, Jing Hu, Lei Wang et al. (Microsoft) | **ICLR 2024** | https://arxiv.org/abs/2307.06945 |
| 39 | **Function Vectors in Large Language Models** | Eric Todd, Millicent L. Li, Arnab Sen Sharma et al. (Northeastern) | **ICLR 2024** | https://arxiv.org/abs/2310.15213 |
