# RAG (Retrieval-Augmented Generation)

---

## 1. 기법의 정의

Retrieval-Augmented Generation(RAG)은 대규모 언어 모델(LLM)의 생성 과정에 외부 지식 저장소로부터의 검색(retrieval)을 결합하는 기법이다. 구체적으로, 주어진 입력 쿼리 $q$에 대해 검색기(retriever) $R$이 외부 코퍼스 $\mathcal{C}$에서 관련 문서 집합 $\mathcal{D} = \{d_1, d_2, \ldots, d_k\}$를 추출하고, 생성기(generator) $G$가 원래 쿼리와 검색된 문서를 조건으로 하여 응답 $y$를 생성하는 구조이다:

$$
p(y | q) = \sum_{d \in \mathcal{D}} p(d | q) \cdot p(y | q, d)
$$

여기서 $p(d | q)$는 검색기의 문서 관련도 분포이며, $p(y | q, d)$는 생성기의 조건부 생성 확률이다. RAG의 핵심 설계 원리는 parametric memory(모델 파라미터에 인코딩된 지식)와 non-parametric memory(외부 문서 저장소)를 분리함으로써, 모델 재학습 없이 지식을 갱신할 수 있도록 하는 것이다.

---

## 2. 기존 기법의 한계와 RAG 등장 배경

### 2.1 Parametric-only 모델의 근본적 한계

사전 학습만 수행한 LLM은 학습 데이터에 포함된 지식을 파라미터에 암묵적으로 저장한다. 이 방식에는 다음과 같은 구조적 문제가 존재한다:

1. **지식 정태성(Knowledge Staleness)**: 학습 데이터의 마감 시점 이후 발생한 사실을 반영할 수 없다. GPT-4의 학습 데이터 마감일은 2023년 4월이며, 이후의 사건에 대해서는 정확한 응답이 불가능하다.

2. **환각(Hallucination)**: 모델은 학습 분포 내에서 통계적으로 그럴듯한 토큰 시퀀스를 생성하므로, 사실과 무관한 내용을 확신 있게 출력하는 현상이 발생한다. TruthfulQA 벤치마크에서 GPT-3 (175B)는 58.0%의 진실성(truthfulness)만을 달성하였다 (Lin et al., 2022).

3. **출처 불투명성**: 생성된 응답이 어떤 학습 데이터에 기반하는지 추적이 불가능하여, 사실 검증(fact verification)이 구조적으로 어렵다.

4. **도메인 특화 비용**: 새로운 도메인 지식을 반영하려면 전체 모델을 Fine-tuning해야 하며, 70B 모델 기준 약 140GB 이상의 GPU VRAM과 수십 시간의 학습이 필요하다.

### 2.2 Open-domain QA에서의 검색-생성 분리 문제

RAG 이전의 Open-domain QA 시스템은 검색기와 생성기(또는 리더)가 독립적으로 학습되었다. Chen et al. (2017)의 DrQA는 TF-IDF 기반 검색기와 RNN 기반 리더를 결합하였으나, 두 모듈이 서로의 그래디언트를 공유하지 않았다. 이로 인해 검색기는 리더가 실제로 필요로 하는 문서가 아닌, 표면적 키워드 매칭에 기반한 문서를 반환하는 문제가 발생하였다.

**구체적 성능 한계**: DrQA는 SQuAD-Open에서 Exact Match(EM) 29.8%를 달성하였다. 검색 단계에서 정답이 포함된 문서를 상위 5개 내에 반환하는 비율(Recall@5)이 약 77.8%에 불과하여, 리더의 성능 상한이 검색기에 의해 제약되는 구조적 병목이 존재하였다.

### 2.3 Sparse Retrieval의 의미적 한계

BM25로 대표되는 sparse retrieval 기법은 토큰의 정확한 매칭(exact match)에 의존한다. BM25의 점수 함수는 다음과 같다:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

여기서 $f(t, d)$는 문서 $d$에서 토큰 $t$의 출현 빈도, $k_1$과 $b$는 하이퍼파라미터(통상 $k_1 = 1.2$, $b = 0.75$)이다. 이 방식은 어휘 불일치(lexical mismatch) 문제를 본질적으로 해결할 수 없다. 예를 들어, "세계에서 가장 큰 도시"라는 쿼리와 "인구 기준 최대 도시"라는 문서 사이의 의미적 동일성을 포착하지 못한다.

---

## 3. 주요 기법 상세

### 3.1 Dense Passage Retrieval (DPR)

**해결한 문제**: BM25의 어휘 불일치 문제를 신경망 기반 밀집 표현(dense representation)으로 해결하였다.

Karpukhin et al. (2020)은 쿼리 인코더 $E_Q$와 문서 인코더 $E_D$를 별도의 BERT 모델로 구성하고, 내적(inner product) 유사도를 학습 목표로 사용하였다:

$$
\text{sim}(q, d) = E_Q(q)^\top E_D(d)
$$

학습은 in-batch negative sampling을 포함한 contrastive loss로 수행된다:

$$
\mathcal{L} = -\log \frac{e^{\text{sim}(q_i, d_i^+)}}{e^{\text{sim}(q_i, d_i^+)} + \sum_{j=1}^{n} e^{\text{sim}(q_i, d_j^-)}}
$$

여기서 $d_i^+$는 양성 문서, $d_j^-$는 음성 문서(동일 배치 내 다른 쿼리의 양성 문서를 음성으로 재활용)이다.

**실험 결과**: Natural Questions(NQ) 데이터셋에서 DPR은 Top-20 Recall 78.4%를 달성하여 BM25의 59.1%를 19.3%p 상회하였다. 최종 QA 성능(EM)은 DPR+리더 조합이 41.5%로, BM25+리더의 32.6%보다 8.9%p 향상되었다.

**잔존 문제**: DPR은 쿼리와 문서를 각각 단일 벡터로 압축하므로, 토큰 수준의 세밀한 상호작용(fine-grained interaction) 정보가 손실된다. 또한 검색기와 생성기가 여전히 분리 학습되어, end-to-end 최적화가 불가능하였다.

### 3.2 RAG (Lewis et al., 2020)

**해결한 문제**: 검색기와 생성기를 단일 프레임워크로 통합하여 end-to-end 학습이 가능하도록 하였다.

Lewis et al. (2020)은 두 가지 변형을 제안하였다:

**RAG-Sequence**: 하나의 검색된 문서로 전체 시퀀스를 생성한 뒤, 문서별 확률을 주변화(marginalize)한다:

$$
p_{\text{RAG-Seq}}(y | q) = \sum_{d \in \text{top-}k(p(\cdot | q))} p(d | q) \prod_{i=1}^{N} p(y_i | q, d, y_{1:i-1})
$$

**RAG-Token**: 각 토큰 생성 시점마다 서로 다른 문서를 참조할 수 있다:

$$
p_{\text{RAG-Token}}(y | q) = \prod_{i=1}^{N} \sum_{d \in \text{top-}k(p(\cdot | q))} p(d | q) \cdot p(y_i | q, d, y_{1:i-1})
$$

검색기는 DPR을 사용하고, 생성기는 BART-large (400M)를 사용하였다. 검색기의 문서 인코더($E_D$)는 고정하되 쿼리 인코더($E_Q$)는 생성기와 함께 end-to-end로 학습하였다.

**실험 결과**: Natural Questions에서 RAG-Token은 EM 44.5%를 달성하여, 당시 SOTA인 DPR+리더(41.5%)를 3.0%p 상회하였다. 특히 Jeopardy Question Generation 태스크에서 RAG는 factual correctness 측면에서 순수 BART 대비 유의미한 향상을 보였다.

**잔존 문제**: (1) 모든 쿼리에 대해 무조건 검색을 수행하므로, 모델이 이미 파라미터에 보유한 지식에 대해서도 불필요한 검색 비용이 발생한다. (2) 검색된 문서의 품질을 평가하는 메커니즘이 없어, 노이즈가 포함된 문서가 생성 품질을 저하시킬 수 있다.

### 3.3 RETRO (Borgeaud et al., 2022)

**해결한 문제**: 사전 학습 단계에서부터 검색을 통합하여, 동일 파라미터 규모 대비 성능을 향상시켰다.

RETRO(Retrieval-Enhanced Transformer)는 입력 시퀀스를 $l$개의 청크 $C_1, C_2, \ldots, C_l$로 분할하고, 각 청크에 대해 외부 코퍼스에서 $k$개의 이웃(nearest neighbor)을 검색한다. 검색된 이웃은 Chunked Cross-Attention(CCA) 레이어를 통해 디코더에 주입된다:

$$
\text{CCA}(H, \text{Ret}) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

여기서 $Q$는 현재 청크의 히든 상태에서 생성되고, $K$와 $V$는 검색된 이웃의 인코딩에서 생성된다.

**실험 결과**: RETRO (7.5B)는 2조(2T) 토큰의 외부 코퍼스를 활용하여, 25배 큰 Jurassic-1 (178B)과 동등한 perplexity를 달성하였다. The Pile 벤치마크에서 RETRO (7.5B)는 baseline Transformer (7.5B) 대비 perplexity를 약 10% 감소시켰다.

**잔존 문제**: 사전 학습 단계에서 검색이 통합되므로, 기존에 학습 완료된 LLM에 적용할 수 없다. 또한 검색 인덱스의 크기가 코퍼스에 비례하여 증가하므로, 2T 토큰 규모에서 인덱스 저장에 약 178GB의 디스크 공간이 필요하다.

### 3.4 Atlas (Izacard et al., 2023)

**해결한 문제**: 검색기와 생성기를 동시에(jointly) 학습하면서, few-shot 학습 능력을 극대화하였다.

Atlas는 Contriever(사전 학습된 dense retriever)와 Fusion-in-Decoder(FiD) 아키텍처를 결합하였다. 핵심 기여는 검색기 학습을 위한 네 가지 attention distillation 기법의 비교 분석이다:

1. **ADist (Attention Distillation)**: 생성기의 cross-attention 가중치를 검색기의 학습 신호로 사용
2. **EMDR²**: Expectation-Maximization 프레임워크에서 검색 문서를 잠재 변수로 처리
3. **PDist (Perplexity Distillation)**: 각 문서를 제거했을 때의 perplexity 변화를 검색기 학습에 활용
4. **LOOP**: 생성기의 출력을 기반으로 검색기를 반복 갱신

**실험 결과**: Atlas (11B = Contriever + FiD-T5-XL)는 Natural Questions에서 64개의 학습 예시만으로 EM 42.4%를 달성하였다. 이는 540B 파라미터의 PaLM이 64-shot에서 달성한 39.6%를 상회하는 결과이다. KILT 벤치마크에서도 5개 태스크 중 4개에서 SOTA를 달성하였다.

**잔존 문제**: 검색기와 생성기의 동시 학습은 학습 안정성(training stability)이 낮고, 검색 인덱스를 주기적으로 재구축(re-index)해야 하는 비용이 발생한다.

### 3.5 ColBERT 및 Late Interaction 기반 검색

**해결한 문제**: 단일 벡터 표현(single-vector representation)의 정보 손실 문제를 토큰 수준의 상호작용으로 해결하였다.

Khattab & Zaharia (2020)의 ColBERT는 쿼리와 문서의 각 토큰을 개별 벡터로 인코딩한 뒤, MaxSim 연산으로 유사도를 계산한다:

$$
\text{ColBERT}(q, d) = \sum_{i \in |q|} \max_{j \in |d|} E_Q(q)_i^\top E_D(d)_j
$$

이 방식은 cross-encoder 수준의 정확도를 유지하면서도, 문서 인코딩을 오프라인으로 사전 계산할 수 있어 검색 속도를 확보한다.

**실험 결과**: MS MARCO Passage Ranking에서 ColBERT는 MRR@10 36.0%를 달성하여, 단일 벡터 DPR(MRR@10 31.1%)을 4.9%p 상회하였다. ColBERTv2(Santhanam et al., 2022)는 잔차 압축(residual compression)을 도입하여 인덱스 크기를 6-10배 줄이면서 성능을 유지하였다.

**잔존 문제**: 토큰별 벡터를 저장하므로 인덱스 크기가 단일 벡터 방식 대비 128배(토큰 수 비례) 크다. ColBERTv2의 압축 이후에도 BM25 인덱스 대비 수십 배 크다.

### 3.6 Self-RAG (Asai et al., 2023)

**해결한 문제**: 모든 쿼리에 대해 무조건 검색을 수행하는 기존 RAG의 비효율성과, 검색 결과의 품질 검증 부재를 해결하였다.

Self-RAG는 LLM이 네 가지 특수 리플렉션 토큰(reflection token)을 생성하도록 학습된다:

1. **[Retrieve]**: 검색이 필요한지 판단 (yes/no/continue)
2. **[IsREL]**: 검색된 문서가 쿼리와 관련 있는지 판단 (relevant/irrelevant)
3. **[IsSUP]**: 생성된 응답이 검색 문서에 의해 지지되는지 판단 (fully/partially/no support)
4. **[IsUSE]**: 생성된 응답의 전반적 유용성 판단 (1-5 등급)

학습 과정은 다음과 같다: (1) GPT-4를 사용하여 리플렉션 토큰의 레이블을 생성(critic model 학습 데이터 구축), (2) 학습된 critic model로 대규모 코퍼스에 리플렉션 토큰을 자동 부착, (3) 리플렉션 토큰이 포함된 데이터로 generator LLM을 학습.

**추론 알고리즘**:
```
Input: 쿼리 q
1. Generator가 [Retrieve] 토큰 생성
2. IF [Retrieve] = yes:
     검색기로 Top-K 문서 검색
     각 문서에 대해:
       - 응답 세그먼트 생성
       - [IsREL], [IsSUP] 토큰으로 품질 평가
     최고 점수 세그먼트 선택
3. ELSE:
     검색 없이 직접 생성
4. [IsUSE] 토큰으로 최종 품질 평가
Output: 응답 y, 리플렉션 토큰 시퀀스
```

**실험 결과**: Self-RAG (LLaMA-2 7B 기반)는 PopQA에서 accuracy 54.9%를 달성하여, ChatGPT의 29.3%와 기존 RAG(LLaMA-2 7B + retrieval)의 42.8%를 각각 25.6%p, 12.1%p 상회하였다. ASQA에서는 citation precision 68.4%를 달성하여 기존 RAG 대비 20%p 이상 향상되었다.

**잔존 문제**: 리플렉션 토큰 학습을 위해 GPT-4 기반의 레이블링이 필요하여 학습 비용이 높다. 또한 추론 시 여러 후보 세그먼트를 병렬 생성하고 평가해야 하므로, 지연 시간이 기본 RAG 대비 2-3배 증가한다.

### 3.7 Corrective RAG (CRAG)

**해결한 문제**: 검색된 문서의 품질이 낮을 때의 대응 전략이 부재한 문제를 해결하였다.

Yan et al. (2024)의 CRAG는 경량 T5-large 기반 검색 평가기(retrieval evaluator)를 도입하여, 검색 결과에 대해 세 가지 판정을 수행한다:

1. **Correct**: 관련 문서가 충분 → 지식 정제(knowledge refinement) 후 생성
2. **Incorrect**: 관련 문서가 부재 → 웹 검색으로 대체
3. **Ambiguous**: 관련성이 불확실 → 내부 지식과 웹 검색을 병합

지식 정제 과정에서는 검색된 문서를 세분화된 지식 단위(knowledge strip)로 분해하고, 각 단위의 관련성을 개별 평가하여 무관한 정보를 필터링한다.

**실험 결과**: CRAG는 PopQA에서 Self-RAG 대비 accuracy 1.8%p 향상을 달성하였으며, 특히 검색 품질이 낮은 long-tail 쿼리에서 5.3%p의 향상을 보였다.

**잔존 문제**: 웹 검색 폴백(fallback) 전략은 외부 검색 엔진(Bing, Google)에 의존하므로, 오프라인 환경에서 사용이 제한된다.

### 3.8 Graph RAG (Edge et al., 2024)

**해결한 문제**: 벡터 검색 기반 RAG는 개별 청크를 독립적으로 취급하여, 문서 컬렉션 전체에 걸친 글로벌 질의(예: "이 데이터셋의 주요 주제는 무엇인가?")에 응답할 수 없는 문제를 해결하였다.

Graph RAG의 인덱싱 파이프라인은 다음과 같다:

```
1. 소스 문서 → 텍스트 청크 분할
2. LLM으로 각 청크에서 엔티티(entity)와 관계(relation) 추출
3. 엔티티-관계 그래프 구축
4. Leiden 알고리즘으로 커뮤니티 탐지 (다수의 계층적 수준에서)
5. 각 커뮤니티에 대해 LLM으로 요약(community summary) 생성
```

쿼리 시에는 관련 커뮤니티 요약을 검색하여 LLM에 제공한다. 이 방식은 map-reduce 패턴을 사용한다: 각 커뮤니티 요약에 대해 부분 응답을 생성(map)한 후, 이를 통합하여 최종 응답을 생성(reduce)한다.

**실험 결과**: Edge et al. (2024)은 Podcast Transcripts와 News Articles 데이터셋에서 평가하였다. 글로벌 질의에 대해 Graph RAG는 기본 RAG 대비 comprehensiveness에서 72% win rate, diversity에서 83% win rate를 달성하였다. 그러나 로컬 질의(특정 엔티티에 대한 질의)에서는 기본 RAG와 유사하거나 약간 낮은 성능을 보였다.

**잔존 문제**: (1) 그래프 구축에 다수의 LLM API 호출이 필요하여 비용이 높다 (약 $0.5-2.0/1000 문서). (2) 문서 업데이트 시 그래프의 점진적 갱신(incremental update)이 어렵다. (3) 엔티티/관계 추출의 정확도가 LLM 능력에 의존한다.

### 3.9 FLARE (Forward-Looking Active REtrieval)

**해결한 문제**: 긴 응답 생성 시, 단일 검색으로는 전체 응답에 필요한 정보를 확보할 수 없는 문제를 해결하였다.

Jiang et al. (2023)의 FLARE는 생성 과정에서 모델의 confidence가 낮아지면 능동적으로 추가 검색을 수행한다. 구체적으로, 다음 문장을 임시 생성(lookahead)하고, 생성된 토큰의 확률이 임계값 $\theta$ 이하인 경우 해당 문장을 쿼리로 변환하여 검색을 수행한다:

$$
\text{if } \min_{t \in s_{i+1}} p(t) < \theta, \text{ then retrieve using } s_{i+1} \text{ as query}
$$

**실험 결과**: 장문 생성 태스크에서 FLARE는 single-retrieval RAG 대비 FactScore를 평균 7.2%p 향상시켰다.

### 3.10 Agentic RAG

**해결한 문제**: 단일 검색-생성 파이프라인으로는 복잡한 다단계 추론 질의(multi-hop question)를 처리할 수 없는 문제를 해결하였다.

Agentic RAG는 LLM을 에이전트(agent)로 활용하여, 검색 전략을 동적으로 수립하고 실행한다. 핵심 구성 요소는 다음과 같다:

1. **Router Agent**: 쿼리를 분석하여 적절한 도구(벡터 검색, 웹 검색, SQL 쿼리, API 호출 등)를 선택
2. **Query Planner**: 복잡한 쿼리를 하위 쿼리로 분해
3. **Retrieval Agent**: 하위 쿼리별로 최적의 검색 전략을 실행
4. **Synthesis Agent**: 수집된 정보를 통합하여 최종 응답을 생성

대표적 구현으로는 LangChain의 ReAct Agent 기반 RAG, LlamaIndex의 SubQuestion Query Engine 등이 있다. Baek et al. (2024)의 연구에서는 Agentic RAG가 HotpotQA에서 multi-hop 질의에 대해 기존 RAG 대비 F1 12.5%p 향상을 보고하였다.

**잔존 문제**: 에이전트의 다단계 도구 호출로 인해 지연 시간이 10-30초로 증가하며, 도구 선택 오류가 전파(error propagation)될 수 있다.

---

## 4. 핵심 구성 요소의 기술적 상세

### 4.1 청킹(Chunking) 전략

문서를 벡터 저장소에 인덱싱하기 위해서는 적절한 단위로 분할해야 한다. 청크 크기는 검색 정밀도와 컨텍스트 보존 사이의 트레이드오프를 결정한다.

| 전략 | 설명 | 청크 크기 | 장점 | 단점 |
|------|------|-----------|------|------|
| **Fixed-size** | 고정 토큰 수로 분할, 오버랩 포함 | 256-512 토큰, 오버랩 10-20% | 구현 단순, 범용적 | 의미 단위 무시 |
| **Recursive** | 구분자 계층(\n\n → \n → . → " ")에 따라 재귀 분할 | 가변 | 문단 경계 존중 | 구분자 설계 필요 |
| **Semantic** | 연속 문장의 임베딩 유사도가 임계값 이하로 하락하는 지점에서 분할 | 가변 | 의미적 일관성 최대화 | 임베딩 연산 비용 |
| **Parent-Child** | 큰 청크(parent)와 작은 청크(child) 이중 구조 | Parent: 2048, Child: 256 | 검색 정밀도 + 컨텍스트 보존 | 인덱스 복잡도 증가 |
| **Agentic (Proposition)** | LLM으로 각 문장을 독립적 명제(proposition)로 변환 | 1 명제 | 최고 정밀도 | LLM 호출 비용 |

Yepes et al. (2024)의 실험에 따르면, semantic chunking은 fixed-size chunking 대비 retrieval recall을 평균 8.3%p 향상시키지만, 청크 생성 시간이 15배 증가한다.

### 4.2 임베딩 모델

검색 품질은 임베딩 모델의 표현 능력에 직접적으로 의존한다. 주요 모델의 성능 비교는 다음과 같다 (MTEB 벤치마크, 2024년 3월 기준):

| 모델 | 차원 | 파라미터 | MTEB Avg | 학습 방식 |
|------|------|----------|----------|-----------|
| **text-embedding-3-large** (OpenAI) | 3072 | 비공개 | 64.6 | 비공개 |
| **voyage-large-2** (Voyage AI) | 1536 | 비공개 | 63.2 | 비공개 |
| **E5-mistral-7b-instruct** (Microsoft) | 4096 | 7B | 66.6 | Instruction-tuned contrastive |
| **GTE-Qwen2-7B-instruct** (Alibaba) | 3584 | 7B | 70.2 | Multi-stage contrastive |
| **BGE-M3** (BAAI) | 1024 | 568M | 59.7 | Multi-lingual, multi-granularity |
| **NV-Embed-v2** (NVIDIA) | 4096 | 7B | 72.3 | Latent attention + two-stage tuning |

임베딩 모델의 학습은 통상 두 단계로 수행된다: (1) 대규모 약한 감독 쌍(weakly supervised pairs, 예: 제목-본문)으로 contrastive pre-training, (2) 소규모 고품질 쌍(human-annotated)으로 fine-tuning. Wang et al. (2024)의 E5-mistral은 synthetic data generation을 활용하여 GPT-4로 학습 데이터를 생성하는 방식을 제안하였다.

### 4.3 벡터 데이터베이스

벡터 데이터베이스는 고차원 벡터의 근사 최근접 이웃 검색(Approximate Nearest Neighbor, ANN)을 수행한다. 주요 ANN 알고리즘의 비교는 다음과 같다:

| 알고리즘 | 시간 복잡도 | 공간 복잡도 | 특징 |
|----------|-------------|-------------|------|
| **HNSW** (Hierarchical NSW) | $O(\log n)$ | $O(n \cdot M)$ | 높은 recall, 메모리 집약적 |
| **IVF-PQ** (Inverted File + Product Quantization) | $O(n / n_{\text{probe}})$ | $O(n \cdot m)$ | 메모리 효율적, recall 약간 낮음 |
| **ScaNN** (Google) | $O(\sqrt{n})$ | $O(n \cdot d)$ | anisotropic quantization |
| **DiskANN** (Microsoft) | $O(\log n)$ | 디스크 기반 | 10억+ 벡터 규모 가능 |

주요 벡터 데이터베이스 솔루션:

| 솔루션 | 유형 | ANN 알고리즘 | 최대 벡터 수 | 메타데이터 필터링 |
|--------|------|-------------|-------------|-------------------|
| **FAISS** (Meta) | 라이브러리 | IVF-PQ, HNSW | 10억+ | 미지원 (직접 구현 필요) |
| **Pinecone** | 관리형 SaaS | 독자 알고리즘 | 수십억 | 지원 |
| **Weaviate** | 오픈소스 DB | HNSW | 수억 | 지원 |
| **Milvus** | 오픈소스 DB | IVF, HNSW, DiskANN | 수십억 | 지원 |
| **Chroma** | 오픈소스 DB | HNSW | 수백만 | 지원 |
| **Qdrant** | 오픈소스 DB | HNSW | 수억 | 지원 (payload 기반) |
| **pgvector** | PostgreSQL 확장 | IVFFlat, HNSW | 수백만 | SQL 기반 |

### 4.4 Re-ranking

초기 검색(first-stage retrieval)은 속도를 우선하여 bi-encoder를 사용하므로 정밀도가 제한된다. Re-ranking은 cross-encoder를 사용하여 쿼리-문서 쌍의 관련도를 정밀하게 재평가한다:

$$
\text{score}(q, d) = \text{CrossEncoder}([q; \text{[SEP]}; d])
$$

Cross-encoder는 쿼리와 문서를 동시에 입력받아 토큰 간 양방향 어텐션을 수행하므로, bi-encoder보다 높은 정확도를 달성한다. 그러나 모든 문서에 대해 개별적으로 추론해야 하므로 전체 코퍼스 검색에는 사용할 수 없고, 초기 검색 결과(통상 Top-100)에 대해서만 적용한다.

대표적 모델:
- **Cohere Rerank v3**: API 기반, 다국어 지원
- **bge-reranker-v2-m3** (BAAI): 오픈소스, 568M 파라미터
- **RankLLaMA** (Ma et al., 2023): LLaMA 기반 listwise reranker
- **RankGPT** (Sun et al., 2023): GPT-4를 listwise reranker로 활용, TREC-DL 2020에서 nDCG@10 74.5% 달성

### 4.5 Hybrid Search

Dense retrieval과 sparse retrieval의 상호 보완적 특성을 활용하는 방식이다. 일반적으로 Reciprocal Rank Fusion(RRF)으로 결과를 통합한다:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
$$

여기서 $R$은 개별 랭킹 목록의 집합, $r(d)$는 랭킹 목록 $r$에서 문서 $d$의 순위, $k$는 상수(통상 60)이다.

대안으로 점수 정규화 후 가중 합산 방식도 사용된다:

$$
\text{score}(d) = \alpha \cdot \text{dense\_score}(d) + (1 - \alpha) \cdot \text{sparse\_score}(d)
$$

Luo et al. (2023)의 실험에서 hybrid search($\alpha = 0.7$)는 BEIR 벤치마크에서 dense-only 대비 nDCG@10을 평균 3.2%p 향상시켰다.

---

## 5. 기법 진화의 인과적 흐름

다음은 RAG 기법이 진화해 온 인과적 체인을 정리한 것이다. 각 단계에서 이전 기법의 문제가 다음 기법의 설계 동기가 되었다.

### Phase 1: 검색-생성 분리 시대 (2017-2019)

**DrQA (Chen et al., 2017)**
- 문제 정의: 구조화되지 않은 대규모 문서에서 질의에 대한 답변 추출
- 해결: TF-IDF 검색 + RNN 리더의 파이프라인 구조
- 잔존 문제: 검색기와 리더의 분리 학습 → 검색기가 리더에 최적화되지 않음

### Phase 2: Dense Retrieval 도입 (2020)

**DPR (Karpukhin et al., 2020)**
- 문제 정의: TF-IDF/BM25의 어휘 불일치 문제
- 해결: BERT 기반 dual-encoder로 의미적 검색 가능
- 잔존 문제: 검색기와 생성기의 end-to-end 학습 불가

### Phase 3: 검색-생성 통합 (2020-2022)

**RAG (Lewis et al., 2020)**
- 문제 정의: 검색기와 생성기의 분리로 인한 차선 성능
- 해결: 검색 문서를 잠재 변수로 처리, 주변화를 통한 end-to-end 학습
- 잔존 문제: 무조건적 검색 수행, 검색 품질 미검증

**RETRO (Borgeaud et al., 2022)**
- 문제 정의: 사전 학습된 LLM에 검색을 사후 결합하는 것의 한계
- 해결: 사전 학습 단계에서 검색 통합, Chunked Cross-Attention
- 잔존 문제: 기존 LLM에 적용 불가, 처음부터 학습 필요

**Atlas (Izacard et al., 2023)**
- 문제 정의: 검색기와 생성기의 동시 최적화 + few-shot 능력
- 해결: Contriever + FiD의 joint training, attention distillation
- 잔존 문제: 학습 불안정성, 인덱스 재구축 비용

### Phase 4: 적응적 검색 (2023)

**FLARE (Jiang et al., 2023)**
- 문제 정의: 장문 생성 시 단일 검색의 한계
- 해결: 생성 confidence 기반 능동적 반복 검색
- 잔존 문제: 검색 시점 결정의 정확도 의존

**Self-RAG (Asai et al., 2023)**
- 문제 정의: 불필요한 검색 수행 + 검색 품질 미검증
- 해결: 리플렉션 토큰으로 검색 필요성, 관련성, 지지도를 자체 평가
- 잔존 문제: 학습 비용 높음 (GPT-4 레이블링 필요), 추론 지연 증가

### Phase 5: 검색 품질 보정 (2024)

**CRAG (Yan et al., 2024)**
- 문제 정의: 검색 결과의 품질이 낮을 때의 대응 전략 부재
- 해결: 검색 평가기 + 지식 정제 + 웹 검색 폴백
- 잔존 문제: 외부 검색 엔진 의존

### Phase 6: 구조적 지식 활용 (2024)

**Graph RAG (Edge et al., 2024)**
- 문제 정의: 벡터 검색의 청크 독립성 가정 → 글로벌 질의 불가
- 해결: 지식 그래프 + 커뮤니티 탐지 + 계층적 요약
- 잔존 문제: 그래프 구축 비용, 실시간 갱신 어려움

### Phase 7: 에이전트 기반 동적 검색 (2024-)

**Agentic RAG**
- 문제 정의: 단일 파이프라인으로 복잡한 다단계 추론 불가
- 해결: LLM 에이전트가 도구 선택, 쿼리 분해, 반복 검색을 동적으로 수행
- 잔존 문제: 지연 시간 증가, 에이전트 오류 전파, 비용 증가

### 현재 연구 방향 (2025-2026)

1. **Speculative RAG** (He et al., 2024): 소형 전문 모델이 다수의 초안을 병렬 생성하고, 대형 모델이 검증하는 방식으로 지연 시간을 감소시킨다.
2. **Long-context LLM vs. RAG**: Gemini 1.5 Pro (1M 컨텍스트)와 같은 장문 컨텍스트 모델이 RAG를 대체할 수 있는지에 대한 논쟁이 진행 중이다. Xu et al. (2024)는 LOFT 벤치마크에서 장문 컨텍스트 모델이 RAG와 동등한 성능을 보이지만, 비용이 10-100배 높음을 보고하였다.
3. **Multimodal RAG**: 텍스트뿐 아니라 이미지, 테이블, 차트 등 다양한 모달리티의 문서를 통합 검색하는 연구가 활발하다.
4. **RAG의 Faithfulness 향상**: 검색된 문서와 생성된 응답 간의 일관성(faithfulness)을 보장하기 위한 constrained decoding, attribution 기법이 연구되고 있다.

---

## 6. 참고 논문

| # | 논문 제목 | 저자 | 학회/출처 | 링크 |
|---|-----------|------|-----------|------|
| 1 | Reading Wikipedia to Answer Open-Domain Questions (DrQA) | Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes | **ACL 2017** | https://arxiv.org/abs/1704.00051 |
| 2 | Dense Passage Retrieval for Open-Domain Question Answering (DPR) | Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih | **EMNLP 2020** | https://arxiv.org/abs/2004.04906 |
| 3 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, Douwe Kiela | **NeurIPS 2020** | https://arxiv.org/abs/2005.11401 |
| 4 | Improving Language Models by Retrieving from Trillions of Tokens (RETRO) | Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, Laurent Sifre | **ICML 2022** | https://arxiv.org/abs/2112.04426 |
| 5 | Atlas: Few-shot Learning with Retrieval Augmented Language Models | Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Tim Rocktaschel, Sebastian Riedel, Douwe Kiela | **JMLR 2023** | https://arxiv.org/abs/2208.03299 |
| 6 | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | Omar Khattab, Matei Zaharia | **SIGIR 2020** | https://arxiv.org/abs/2004.12832 |
| 7 | ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction | Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia | **NAACL 2022** | https://arxiv.org/abs/2112.01488 |
| 8 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi | **ICLR 2024** | https://arxiv.org/abs/2310.11511 |
| 9 | Corrective Retrieval Augmented Generation (CRAG) | Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling | arXiv 2024 | https://arxiv.org/abs/2401.15884 |
| 10 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson | arXiv 2024 | https://arxiv.org/abs/2404.16130 |
| 11 | Active Retrieval Augmented Generation (FLARE) | Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig | **EMNLP 2023** | https://arxiv.org/abs/2305.06983 |
| 12 | REPLUG: Retrieval-Augmented Black-Box Language Models | Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih | **NAACL 2024** | https://arxiv.org/abs/2301.12652 |
| 13 | REALM: Retrieval-Augmented Language Model Pre-Training | Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang | **ICML 2020** | https://arxiv.org/abs/2002.08909 |
| 14 | Fusion-in-Decoder: Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering | Gautier Izacard, Edouard Grave | **EACL 2021** | https://arxiv.org/abs/2007.01282 |
| 15 | TruthfulQA: Measuring How Models Mimic Human Falsehoods | Stephanie Lin, Jacob Hilton, Owain Evans | **ACL 2022** | https://arxiv.org/abs/2109.07958 |
| 16 | Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning | Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave | **TMLR 2022** | https://arxiv.org/abs/2112.09118 |
| 17 | Internet-Augmented Dialogue Generation | Mojtaba Komeili, Kurt Shuster, Jason Weston | **ACL 2022** | https://arxiv.org/abs/2107.07566 |
| 18 | Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP | Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, Matei Zaharia | arXiv 2023 | https://arxiv.org/abs/2212.14024 |
| 19 | RankGPT: Large Language Models are Zero-Shot Rankers | Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren | **EMNLP 2023** | https://arxiv.org/abs/2304.09542 |
| 20 | Retrieval-Augmented Generation for Large Language Models: A Survey | Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang | arXiv 2024 | https://arxiv.org/abs/2312.10997 |
| 21 | When Large Language Models Meet Vector Databases: A Survey | Zhi Jing, Yongye Su, Yikun Han | arXiv 2024 | https://arxiv.org/abs/2402.01763 |
| 22 | E5-Mistral: Text Embeddings by Weakly-Supervised Contrastive Pre-training | Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei | **ACL 2024** | https://arxiv.org/abs/2401.00368 |
| 23 | CRAG: Comprehensive RAG Benchmark | Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziber Liao, Jianpeng Xu, Jun Ma | **KDD 2024** | https://arxiv.org/abs/2406.04744 |
| 24 | Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting | Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-Yu Lee, Tomas Pfister | arXiv 2024 | https://arxiv.org/abs/2407.08223 |
| 25 | Lost in the Middle: How Language Models Use Long Contexts | Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang | **TACL 2024** | https://arxiv.org/abs/2307.03172 |
| 26 | BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu | arXiv 2024 | https://arxiv.org/abs/2402.03216 |
| 27 | Hierarchical Navigable Small World Graphs (HNSW) | Yuri A. Malkov, Dmitry A. Yashunin | **IEEE TPAMI 2020** | https://arxiv.org/abs/1603.09320 |
| 28 | Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE) | Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk | **ICLR 2021** | https://arxiv.org/abs/2007.00808 |
| 29 | FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation | Sebastian Hofstatter, Jiecao Chen, Karthik Raman, Hamed Zamani | **SIGIR 2023** | https://arxiv.org/abs/2209.14290 |
| 30 | Query Rewriting for Retrieval-Augmented Large Language Models | Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan | **EMNLP 2023** | https://arxiv.org/abs/2305.14283 |
| 31 | Benchmarking Large Language Models in Retrieval-Augmented Generation (RGB Benchmark) | Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun | **AAAI 2024** | https://arxiv.org/abs/2309.01431 |
| 32 | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning | **ICLR 2024** | https://arxiv.org/abs/2401.18059 |
| 33 | Seven Failure Points When Engineering a Retrieval Augmented Generation System | Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek | arXiv 2024 | https://arxiv.org/abs/2401.05856 |
| 34 | Improving Text Embeddings with Large Language Models (NV-Embed) | Chankyu Lee, Rajarshi Roy, Menber Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping | arXiv 2024 | https://arxiv.org/abs/2401.17797 |
| 35 | Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? (LOFT) | Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan, Michael Boratko, Yi Luan, Sébastien M. R. Arnold, Vincent Perot, Siddharth Dalmia, Hexiang Hu, Xudong Lin, Panupong Pasupat, Aida Amini, Jeremy R. Cole, Sebastian Riedel, Iftekhar Naim, Ming-Wei Chang, Kelvin Guu | arXiv 2024 | https://arxiv.org/abs/2406.13121 |

---

## 부록: RAG 시스템 설계 시 주요 고려사항

### A.1 Lost in the Middle 현상

Liu et al. (2024)는 LLM이 긴 컨텍스트의 중간 부분에 위치한 정보를 효과적으로 활용하지 못하는 현상을 보고하였다. 20개의 검색 문서를 제공했을 때, 정답이 첫 번째 또는 마지막 위치에 있을 때의 성능이 중간 위치 대비 최대 20%p 높았다. 이는 RAG 시스템에서 검색 문서의 배치 순서가 성능에 유의미한 영향을 미침을 의미한다.

### A.2 RAG 시스템의 7가지 실패 지점

Barnett et al. (2024)은 RAG 시스템의 주요 실패 지점을 다음과 같이 분류하였다:

1. **Missing Content**: 정답이 코퍼스에 존재하지 않음
2. **Missed the Top Ranked Documents**: 정답 문서가 검색되었으나 Top-K 밖에 위치
3. **Not in Context (Consolidation)**: 관련 정보가 여러 문서에 분산
4. **Not Extracted**: 정답이 컨텍스트에 존재하지만 LLM이 추출하지 못함
5. **Wrong Format**: 정답은 맞으나 형식이 부적합
6. **Incorrect Specificity**: 답변의 구체성 수준이 부적절
7. **Incomplete**: 부분적으로만 정확한 응답

### A.3 RAPTOR: 계층적 검색

Sarthi et al. (2024)의 RAPTOR는 문서 청크를 클러스터링한 뒤, 각 클러스터를 LLM으로 요약하여 트리 구조의 인덱스를 구축한다. 리프 노드는 원본 청크이고, 상위 노드는 하위 노드의 요약이다. 질의 시에는 트리의 적절한 수준에서 검색을 수행하여, 세부 사실과 고수준 개요를 모두 검색할 수 있다. QuALITY 벤치마크에서 RAPTOR는 기존 RAG 대비 accuracy를 20%p 이상 향상시켰다.
