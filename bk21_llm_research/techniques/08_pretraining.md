# Pre-training (사전 학습)

---

## 1. 기법의 정의

Pre-training(사전 학습)은 대규모 비지도 텍스트 코퍼스에 대해 자기지도 학습(self-supervised learning) 목적 함수를 최적화함으로써, 범용 언어 표현(universal linguistic representation)을 모델 파라미터에 내재화하는 과정이다. 사전 학습의 핵심 전제는, 충분한 규모의 텍스트에 대해 다음 토큰 예측(next-token prediction) 또는 마스킹 토큰 복원(masked token recovery)을 수행하면, 문법 구조, 의미 관계, 상식 지식, 논리적 추론 패턴 등이 모델의 가중치에 자동으로 인코딩된다는 것이다.

수학적으로, 사전 학습의 목적 함수는 코퍼스 D에 대해 다음과 같이 정의된다:

**Causal Language Modeling (CLM):**
```
L_CLM(θ) = -Σ_{i=1}^{N} Σ_{t=1}^{T_i} log P_θ(x_t^{(i)} | x_{<t}^{(i)})
```

여기서 x^{(i)}는 i번째 문서, x_t^{(i)}는 해당 문서의 t번째 토큰, θ는 모델 파라미터이다. CLM은 자기회귀적(autoregressive) 분해에 기반하며, 현재 시점 이전의 토큰들만을 조건부로 활용하여 다음 토큰의 확률을 최대화한다.

**Masked Language Modeling (MLM):**
```
L_MLM(θ) = -Σ_{i=1}^{N} Σ_{t∈M_i} log P_θ(x_t^{(i)} | x_{\M_i}^{(i)})
```

여기서 M_i는 i번째 문서에서 마스킹된 토큰 인덱스 집합이며, x_{\M_i}는 마스킹되지 않은 토큰들이다. BERT에서는 전체 토큰의 15%를 무작위로 마스킹하며, 마스킹된 위치의 원래 토큰을 양방향 문맥으로부터 복원하도록 학습한다.

**Span Corruption (T5):**
```
L_SC(θ) = -Σ_{i=1}^{N} log P_θ(y^{(i)} | x_{corrupt}^{(i)})
```

여기서 x_{corrupt}는 연속된 토큰 스팬이 단일 센티널 토큰으로 대체된 입력이며, y는 해당 스팬의 원래 토큰 시퀀스이다. 이 방식은 인코더-디코더 아키텍처에서 활용되며, 다양한 길이의 텍스트 생성 능력을 학습한다.

사전 학습은 모델이 특정 태스크에 대한 명시적 레이블 없이도, 텍스트의 통계적 규칙성으로부터 범용적 언어 능력을 획득하도록 하는 기반 단계(foundation stage)이다.

---

## 2. 기존 기법의 한계와 사전 학습 등장 배경

### 2.1 태스크별 개별 학습의 비효율

사전 학습 이전의 NLP 연구에서는 각 태스크(감성 분석, 개체명 인식, 기계 번역 등)마다 별도의 모델을 처음부터(from scratch) 학습하는 것이 일반적이었다. 이 접근법은 다음과 같은 근본적 한계를 가진다:

1. **데이터 효율성의 부재**: 각 태스크의 레이블 데이터는 수천~수만 건에 불과한 경우가 대부분이다. 이 규모로는 심층 신경망이 요구하는 풍부한 언어 표현을 학습하기 불가능하다.
2. **지식 재활용 불가**: 감성 분석 모델이 학습한 언어 지식은 기계 번역 모델에 전이되지 않는다. 동일한 언어 현상(구문 구조, 의미 관계 등)을 매번 새로 학습해야 하므로 연산 자원이 중복 소모된다.
3. **컴퓨터 비전과의 격차**: 컴퓨터 비전 분야에서는 ImageNet 사전 학습 후 전이 학습(transfer learning)이 2012년 AlexNet 이후 표준 파이프라인으로 정착되어 있었다. NLP에서는 Word2Vec 수준의 정적 임베딩만이 전이 가능한 유일한 자원이었다.

### 2.2 정적 단어 임베딩의 한계

Word2Vec (Mikolov et al., 2013)과 GloVe (Pennington et al., 2014)는 단어를 고정된 벡터로 표현하는 정적 임베딩(static embedding) 방식이다. 이 접근법의 핵심 문제는 다의어(polysemy)를 처리할 수 없다는 것이다. 예를 들어, "bank"는 "은행"과 "강둑"이라는 서로 다른 의미를 가지지만, 정적 임베딩에서는 단일 벡터로 표현된다. 이는 문맥 의존적 의미(contextualized meaning)를 포착할 수 없는 구조적 한계이다.

**문제**: 정적 임베딩은 토큰 수준의 표현만 제공할 뿐, 문장 수준의 의미 이해를 위한 상위 구조 학습은 여전히 태스크별로 수행해야 한다.

### 2.3 문맥 의존 표현의 등장과 아키텍처 병목

ELMo (Peters et al., 2018)는 양방향 LSTM(BiLSTM)을 이용하여 문맥 의존적 단어 표현을 생성하는 최초의 사전 학습 모델이다. ELMo는 동일한 단어가 서로 다른 문맥에서 상이한 벡터를 가지도록 함으로써, 정적 임베딩의 다의어 문제를 해결하였다.

**새로운 문제**: LSTM 기반 아키텍처는 시퀀스 길이에 대해 순차적(sequential) 연산이 필수적이므로, GPU 병렬화에 근본적 한계가 존재한다. 이로 인해 모델 규모를 확장(scaling)하기 어려우며, 긴 시퀀스에 대한 장거리 의존성(long-range dependency) 포착 능력도 제한적이다. Vaswani et al. (2017)의 Transformer 아키텍처가 이 병목을 해결하면서, 사전 학습의 패러다임이 근본적으로 전환된다.

---

## 3. 주요 기법 상세

### 3.1 Word2Vec에서 Transformer까지: 표현 학습의 진화

#### 3.1.1 Word2Vec (Mikolov et al., 2013)

Word2Vec은 Skip-gram과 CBOW(Continuous Bag of Words) 두 가지 학습 목표를 제안하였다.

**Skip-gram 목적 함수:**
```
L_SG = -Σ_{t=1}^{T} Σ_{-c≤j≤c, j≠0} log P(w_{t+j} | w_t)
```

여기서 c는 윈도우 크기이다. 중심 단어로부터 주변 단어를 예측하는 과정에서 단어 간 의미적 관계가 벡터 공간에 인코딩된다. 유명한 "king - man + woman ≈ queen" 관계는 이 벡터 공간의 선형 구조를 보여주는 사례이다.

**문제**: 단어당 하나의 고정 벡터만 존재하며, 문맥 정보가 반영되지 않는다.

#### 3.1.2 ELMo (Peters et al., 2018)

ELMo는 2층 BiLSTM으로 순방향/역방향 언어 모델을 동시에 학습한다. 각 토큰의 표현은 모든 BiLSTM 층의 출력을 가중합(weighted sum)하여 생성된다.

```
ELMo_k = γ Σ_{j=0}^{L} s_j · h_{k,j}
```

여기서 h_{k,j}는 j번째 층에서의 k번째 토큰 hidden state, s_j는 softmax 정규화된 층별 가중치, γ는 스칼라 스케일링 팩터이다.

**문제**: BiLSTM의 순차적 연산으로 인해 학습 병렬화가 불가능하며, 모델 규모 확장에 한계가 존재한다.

#### 3.1.3 Transformer 아키텍처의 등장 (Vaswani et al., 2017)

Transformer는 Self-Attention 메커니즘을 통해 시퀀스 내 모든 위치 쌍 간의 관계를 병렬적으로 계산한다.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

여기서 Q, K, V는 각각 Query, Key, Value 행렬이며, d_k는 Key의 차원이다. 이 연산은 O(n^2 · d)의 계산 복잡도를 가지지만, 행렬 곱셈으로 구현되므로 GPU에서 완전히 병렬화 가능하다. 이로 인해 LSTM 대비 학습 속도가 수 배~수십 배 향상되었으며, 대규모 사전 학습의 실현 가능성이 열렸다.

**새로운 문제**: Self-Attention의 O(n^2) 메모리 복잡도는 긴 시퀀스 처리의 병목이 되며, 이 문제는 이후 Flash Attention 등의 연구로 이어진다.

### 3.2 사전 학습 아키텍처 패러다임

#### 3.2.1 BERT: Encoder-only 양방향 사전 학습 (Devlin et al., 2019)

BERT는 Transformer Encoder를 기반으로 MLM과 Next Sentence Prediction(NSP) 두 가지 목적 함수를 사용한다. BERT-Base(110M), BERT-Large(340M) 두 가지 규모로 공개되었으며, 11개 NLU 벤치마크에서 동시에 SOTA를 달성하여 사전 학습의 유효성을 입증하였다.

**문제**: Encoder-only 구조는 양방향 문맥을 활용하기에 이해(understanding) 태스크에는 강하지만, 자기회귀적 텍스트 생성(autoregressive generation)이 구조적으로 불가능하다. MLM 목적 함수에서 사용되는 [MASK] 토큰이 추론(inference) 시에는 존재하지 않으므로, 학습-추론 간 불일치(train-test mismatch)도 발생한다.

#### 3.2.2 GPT-2 / GPT-3: Decoder-only 자기회귀 사전 학습

GPT-2 (Radford et al., 2019)는 1.5B 파라미터의 Decoder-only Transformer로, CLM 목적 함수만으로 사전 학습되었다. 핵심 기여는 명시적 fine-tuning 없이도 zero-shot으로 다양한 태스크를 수행할 수 있음을 보인 것이다.

GPT-3 (Brown et al., 2020)는 175B 파라미터로 규모를 확장하여, few-shot in-context learning이라는 창발적(emergent) 능력이 나타남을 입증하였다. GPT-3의 학습에는 약 3.14 × 10^23 FLOPs가 소요되었으며, 이는 300B 토큰에 대한 1회 에폭(epoch) 학습에 해당한다.

**문제**: GPT-3는 175B 파라미터에 300B 토큰만으로 학습되어, 토큰/파라미터 비율이 약 1.7에 불과하다. 이후 Chinchilla 연구에서 이 비율이 최적점(약 20)에 크게 미달한다는 것이 밝혀진다.

#### 3.2.3 T5: Encoder-Decoder 통합 프레임워크 (Raffel et al., 2020)

T5 (Text-to-Text Transfer Transformer)는 모든 NLP 태스크를 "텍스트 입력 → 텍스트 출력" 형식으로 통일하는 프레임워크를 제안하였다. C4(Colossal Clean Crawled Corpus) 데이터셋을 구축하고, Span Corruption 목적 함수를 사용하여 사전 학습한다. T5 논문은 사전 학습의 다양한 설계 선택(아키텍처, 목적 함수, 데이터 크기, 전이 전략 등)에 대한 체계적 비교 실험을 수행한 점에서 참조 가치가 높다.

**문제**: Encoder-Decoder 구조는 추론 시 Encoder와 Decoder 양쪽의 파라미터를 모두 적재해야 하므로, 동일 파라미터 수의 Decoder-only 모델 대비 추론 효율이 낮다. 이후 대규모 LLM 연구에서는 Decoder-only가 사실상 표준(de facto standard)으로 수렴한다.

### 3.3 Scaling Laws: 사전 학습의 정량적 법칙

#### 3.3.1 Kaplan Scaling Laws (Kaplan et al., 2020)

OpenAI의 Kaplan et al.은 모델 크기(N), 데이터셋 크기(D), 연산량(C) 각각에 대해 손실(L)이 멱법칙(power law)을 따른다는 것을 실증적으로 발견하였다.

```
L(N) = (N_c / N)^{α_N},  α_N ≈ 0.076
L(D) = (D_c / D)^{α_D},  α_D ≈ 0.095
L(C) = (C_c / C)^{α_C},  α_C ≈ 0.050
```

여기서 N_c, D_c, C_c는 각각 상수이다. 핵심 결론은 고정된 연산 예산(compute budget) 하에서 모델 크기를 우선적으로 키우는 것이 최적이며, 데이터 크기는 상대적으로 덜 중요하다는 것이었다.

**문제**: 이 결론은 GPT-3의 학습 전략(큰 모델, 적은 데이터)에 직접적 영향을 미쳤으나, 이후 Chinchilla 연구에서 실험 설계의 결함이 지적되며 뒤집어진다.

#### 3.3.2 Chinchilla Scaling Laws (Hoffmann et al., 2022)

DeepMind의 Hoffmann et al.은 400개 이상의 모델(70M~16B)을 학습시키며 Kaplan의 결론을 재검증하였다. 그 결과, 고정 연산 예산 하에서 모델 크기와 데이터 크기는 동등하게 스케일링되어야 한다는 수정된 법칙을 제시하였다.

**Chinchilla 최적 관계:**
```
N_opt ∝ C^{0.50}
D_opt ∝ C^{0.50}
```

즉, 연산 예산이 10배 증가하면 모델 크기와 데이터 크기 모두 약 √10 ≈ 3.16배씩 증가시키는 것이 최적이다. 실용적 근사 규칙은 다음과 같다:

```
최적 토큰 수 ≈ 20 × 파라미터 수
```

이 법칙에 따르면, GPT-3(175B 파라미터, 300B 토큰)는 약 3.5T 토큰으로 학습되었어야 한다. Chinchilla(70B 파라미터, 1.4T 토큰)는 GPT-3보다 작은 모델이지만, 최적 비율로 학습됨으로써 GPT-3를 상회하는 성능을 달성하였다.

**주요 모델의 토큰/파라미터 비율:**

| 모델 | 파라미터 | 학습 토큰 | 토큰/파라미터 비율 | Chinchilla 최적 대비 |
|------|---------|-----------|-------------------|---------------------|
| GPT-3 | 175B | 300B | 1.7 | Under-trained |
| Chinchilla | 70B | 1.4T | 20 | 최적 |
| LLaMA-1 | 65B | 1.4T | 21.5 | 최적 |
| LLaMA-2 | 70B | 2T | 28.6 | 의도적 Over-training |
| LLaMA-3 | 70B | 15T | 214 | 극단적 Over-training |
| Mistral 7B | 7B | 추정 >2T | >285 | 극단적 Over-training |

**새로운 문제와 추세**: Chinchilla 법칙은 "학습 비용 최적화"에 초점을 맞추지만, 실제 배포(deployment) 환경에서는 "추론 비용 최적화"가 더 중요하다. 작은 모델을 Chinchilla 최적점을 초과하여 대량의 데이터로 over-train하면, 학습 비용은 증가하지만 추론 시 더 적은 파라미터로 동등한 성능을 달성할 수 있다. LLaMA 시리즈가 이 전략의 대표적 사례이다.

### 3.4 LLaMA 시리즈: 오픈소스 사전 학습의 전환점

#### 3.4.1 LLaMA-1 (Touvron et al., 2023a)

LLaMA-1은 7B, 13B, 33B, 65B 네 가지 규모로 공개되었으며, Chinchilla-optimal 수준의 토큰 수(1.0~1.4T)로 학습되었다. 핵심 기여는 공개된 데이터만으로 학습하여도 독점 모델(GPT-3, PaLM)에 필적하는 성능을 달성할 수 있음을 입증한 것이다. 아키텍처적으로는 RMSNorm (Pre-Normalization), SwiGLU 활성 함수, RoPE (Rotary Position Embedding)를 채택하였다.

**학습 비용**: LLaMA-65B는 2048개의 A100 80GB GPU로 약 21일간 학습되었으며, 총 연산량은 약 1.4 × 10^24 FLOPs이다.

**문제**: LLaMA-1의 라이선스는 연구 목적에 한정되어 상업적 활용이 불가능하였다.

#### 3.4.2 LLaMA-2 (Touvron et al., 2023b)

LLaMA-2는 학습 토큰을 2T로 증가시키고(Chinchilla 최적의 약 1.4배 over-training), 컨텍스트 길이를 2048에서 4096으로 확장하였다. 상업적 라이선스를 허용하여 오픈소스 LLM 생태계의 실질적 확산을 견인하였다. Grouped Query Attention(GQA)을 채택하여 추론 시 KV-cache 메모리를 절감하였다.

#### 3.4.3 LLaMA-3 (Meta AI, 2024)

LLaMA-3는 8B, 70B, 405B 세 가지 규모로 공개되었으며, 15T+ 토큰으로 학습되었다. 이는 70B 모델 기준 Chinchilla 최적의 약 10배에 해당하는 극단적 over-training이다. 학습에 약 3.08 × 10^25 FLOPs(405B 모델 기준)가 소요되었으며, 16,384개의 H100 GPU가 사용되었다. 128K 토큰의 컨텍스트 길이를 지원하며, 다국어 토크나이저(128K 어휘)를 사용한다.

**학습 인프라 규모**: LLaMA-3 405B의 학습에는 약 30.84M GPU-hours가 소요되었으며, 이는 전력 비용만으로도 수천만 달러에 해당한다.

### 3.5 Mistral: 효율적 아키텍처 혁신 (Jiang et al., 2023)

Mistral 7B는 Sliding Window Attention(SWA)과 Grouped Query Attention(GQA)을 결합하여, 7B 규모에서 LLaMA-2 13B를 상회하는 성능을 달성하였다. SWA는 고정 크기의 윈도우 내에서만 어텐션을 계산하여, 긴 시퀀스에 대한 추론 비용을 O(n × w)로 감소시킨다(w는 윈도우 크기). Mistral의 후속 모델인 Mixtral 8x7B는 Mixture of Experts(MoE)를 도입하여, 총 파라미터 46.7B 중 활성 파라미터 12.9B만으로 추론을 수행한다.

### 3.6 데이터 준비(Data Preparation)

사전 학습 데이터의 품질은 모델 성능을 결정하는 가장 중요한 단일 요소(single most important factor)이다. "Garbage in, garbage out" 원칙이 LLM 사전 학습에서 극단적으로 적용된다.

#### 3.6.1 주요 공개 학습 데이터셋

| 데이터셋 | 규모 | 공개 기관 | 핵심 특징 |
|----------|------|-----------|-----------|
| **The Pile** | 825GB (약 300B 토큰) | EleutherAI | 22개 고품질 소스의 다양성 중심 혼합 |
| **RedPajama v1** | 1.2T 토큰 | Together AI | LLaMA-1 학습 데이터 재현 시도 |
| **RedPajama v2** | 30T+ 원시 토큰 | Together AI | 대규모 품질 신호(quality signals) 제공 |
| **RefinedWeb** | 5T 토큰 | TII (Falcon) | 엄격한 웹 데이터 필터링 |
| **Dolma** | 3T 토큰 | AI2 (Allen Institute) | OLMo 학습용, 완전 공개 파이프라인 |
| **FineWeb** | 15T 토큰 | HuggingFace | 최대 규모 공개 웹 데이터, 체계적 품질 관리 |
| **FineWeb-Edu** | 1.3T 토큰 | HuggingFace | FineWeb에서 교육적 콘텐츠만 추출 |
| **C4** | 750GB | Google | T5 학습용, Common Crawl 기반 정제 |

#### 3.6.2 데이터 전처리 파이프라인

사전 학습 데이터 전처리는 다음 단계로 구성되며, 각 단계는 특정 노이즈 유형을 제거하는 것을 목적으로 한다:

**1단계: URL 및 도메인 필터링**
- 악성 사이트, 성인 콘텐츠, 스팸 사이트의 URL을 블랙리스트 기반으로 제거한다.
- 도메인 수준의 품질 점수를 산출하여 저품질 도메인을 일괄 배제한다.

**2단계: 언어 감지(Language Identification)**
- fastText 기반 언어 분류기를 사용하여 대상 언어 외의 텍스트를 필터링한다.
- 다국어 모델의 경우, 언어별 비율을 의도적으로 조절한다.

**3단계: 휴리스틱 품질 필터링**
- 문서 길이, 단어 수, 특수문자 비율, 반복 n-gram 비율 등의 규칙 기반 필터를 적용한다.
- FineWeb에서는 이러한 휴리스틱 필터만으로도 상당한 성능 향상을 보고하였다.

**4단계: 모델 기반 품질 필터링**
- 사전 학습된 언어 모델의 perplexity를 기준으로 비정상적 텍스트를 제거한다.
- FineWeb-Edu에서는 교육적 가치를 판별하는 분류기를 추가 적용하였다.

**5단계: 중복 제거(Deduplication)**
- **Exact Deduplication**: URL 또는 문서 해시 기반으로 동일 문서를 제거한다.
- **Fuzzy Deduplication**: MinHash + LSH(Locality-Sensitive Hashing)를 사용하여 유사 문서를 제거한다. Jaccard 유사도 임계값(통상 0.7~0.8)을 설정하여 근사적으로 중복을 탐지한다. Lee et al. (2022)의 연구에서는 중복 제거만으로 C4 데이터에서 학습 성능이 유의미하게 향상됨을 보고하였다.

**6단계: PII(개인정보) 제거 및 유해 콘텐츠 필터링**
- 이메일, 전화번호, 주소 등의 개인정보를 정규식 및 NER 모델로 탐지하여 제거 또는 마스킹한다.
- 유해 콘텐츠(혐오 표현, 폭력, 불법 정보 등)를 분류기로 필터링한다.

**7단계: 데이터 혼합(Data Mixing)**
- 서로 다른 도메인(웹, 도서, 코드, 학술 논문, Wikipedia 등)의 비율을 조절한다.
- LLaMA-1에서는 영어 Common Crawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Books 4.5%, ArXiv 2.5%, Stack Exchange 2%의 비율을 사용하였다.
- 최적 혼합 비율은 이론적 근거가 부족하며, 대부분 경험적(empirical) 탐색에 의존한다. DoReMi (Xie et al., 2023)는 프록시 모델을 활용한 자동 데이터 혼합 최적화 기법을 제안하였다.

### 3.7 분산 학습(Distributed Training)

수십~수백 B 파라미터의 LLM을 학습하려면 수천~수만 개의 GPU를 동시에 활용하는 분산 학습이 필수적이다. 단일 GPU의 메모리(80GB, A100/H100 기준)는 7B 모델의 학습 상태(파라미터 + 옵티마이저 + 그래디언트 + 활성값)도 수용하기 어렵다.

#### 3.7.1 Data Parallelism (DP)

가장 기본적인 병렬화 기법이다. 모델 전체를 각 GPU에 복제하고, 미니배치를 GPU 수만큼 분할하여 병렬 처리한 후, 그래디언트를 All-Reduce 연산으로 동기화한다.

**문제**: 모델 전체가 각 GPU에 복제되므로, 단일 GPU 메모리에 모델이 적재되지 않으면 사용 불가능하다.

#### 3.7.2 ZeRO (Zero Redundancy Optimizer) (Rajbhandari et al., 2020)

ZeRO는 Data Parallelism의 메모리 중복 문제를 해결한다. 모델 학습 시 GPU별로 저장되는 상태를 3단계로 분산시킨다.

```
학습 시 GPU당 메모리 (Mixed Precision, AdamW, 모델 파라미터 수 Ψ):
  - 파라미터 (FP16): 2Ψ bytes
  - 그래디언트 (FP16): 2Ψ bytes
  - 옵티마이저 상태 (FP32 사본 + momentum + variance): 12Ψ bytes
  - 총합: 16Ψ bytes per GPU (DP without ZeRO)
```

| ZeRO Stage | 분산 대상 | GPU당 메모리 (N_d개 GPU) |
|------------|-----------|------------------------|
| Stage 1 | 옵티마이저 상태 | 2Ψ + 2Ψ + 12Ψ/N_d |
| Stage 2 | + 그래디언트 | 2Ψ + (2Ψ + 12Ψ)/N_d |
| Stage 3 | + 파라미터 | 16Ψ/N_d |

ZeRO Stage 3에서는 N_d개 GPU 사용 시 GPU당 메모리가 1/N_d로 감소하므로, 이론적으로는 임의 크기의 모델을 학습할 수 있다. 통신 오버헤드는 Stage 1/2에서 DP와 동일(All-Reduce 1회)하며, Stage 3에서는 추가 통신이 발생한다.

#### 3.7.3 FSDP (Fully Sharded Data Parallel)

FSDP는 PyTorch에서 공식 지원하는 ZeRO Stage 3 구현이다. ZeRO의 개념을 PyTorch 네이티브 API로 통합하여, DeepSpeed 없이도 모델 상태 분산이 가능하다. LLaMA-2, LLaMA-3의 학습에 FSDP가 사용되었다.

#### 3.7.4 Tensor Parallelism (TP) (Shoeybi et al., 2019)

Tensor Parallelism은 단일 레이어 내부의 행렬 연산을 여러 GPU에 분산한다. Megatron-LM에서 제안되었으며, 주로 Transformer의 Attention 및 FFN(Feed-Forward Network) 레이어에 적용된다.

**FFN 분할 예시:**
```
Y = GeLU(X · A) · B
→ Column-wise split: A = [A_1, A_2], row-wise split: B = [B_1; B_2]
→ GPU 1: Y_1 = GeLU(X · A_1) · B_1
→ GPU 2: Y_2 = GeLU(X · A_2) · B_2
→ Y = Y_1 + Y_2 (All-Reduce)
```

TP는 레이어 내부에서 병렬화하므로, 단일 레이어의 연산량이 큰 대규모 모델에서 효과적이다. 그러나 레이어 내부에서 All-Reduce 통신이 필요하므로, GPU 간 대역폭이 높은 환경(NVLink, NVSwitch)에서만 효율적이다.

#### 3.7.5 Pipeline Parallelism (PP) (Huang et al., 2019)

Pipeline Parallelism은 모델의 연속된 레이어 그룹을 서로 다른 GPU에 배치한다. GPipe에서 제안되었으며, 미니배치를 마이크로배치(micro-batch)로 분할하여 파이프라인 버블(bubble)을 최소화한다.

**파이프라인 버블 비율:**
```
Bubble ratio = (p - 1) / (m + p - 1)
```

여기서 p는 파이프라인 스테이지 수, m은 마이크로배치 수이다. m >> p이면 버블 비율이 0에 수렴한다.

**문제**: 파이프라인 버블로 인한 GPU 유휴(idle) 시간이 발생하며, 이를 줄이기 위해서는 마이크로배치 수를 증가시켜야 한다. 이는 메모리 사용량 증가와 상충한다.

#### 3.7.6 3D Parallelism (Narayanan et al., 2021)

3D Parallelism은 DP, TP, PP를 동시에 적용하는 기법이다. 대규모 학습에서는 다음과 같이 구성된다:

```
총 GPU 수 = N_dp × N_tp × N_pp

예시 (LLaMA-3 405B):
- N_tp = 8 (노드 내 NVLink 연결 GPU)
- N_pp = 16 (노드 간 파이프라인)
- N_dp = 128 (데이터 병렬)
- 총 GPU: 8 × 16 × 128 = 16,384
```

TP는 고대역폭 노드 내(intra-node) 통신에, PP는 노드 간(inter-node) 통신에, DP는 전체 규모 확장에 각각 활용되는 것이 일반적인 설계 원칙이다.

### 3.8 학습 효율화 기법

#### 3.8.1 Flash Attention (Dao et al., 2022)

표준 Self-Attention 구현은 N×N 크기의 어텐션 행렬을 GPU HBM(High Bandwidth Memory)에 저장해야 하므로, 시퀀스 길이에 대해 O(N^2) 메모리가 필요하다. Flash Attention은 타일링(tiling) 기법을 사용하여 어텐션 연산을 SRAM(on-chip memory)에서 수행함으로써, HBM 접근 횟수를 O(N^2)에서 O(N^2 / M)으로 감소시킨다(M은 SRAM 크기).

**핵심 기법**: Online Softmax 알고리즘을 활용하여, 전체 어텐션 행렬을 명시적으로 구성하지 않고도 정확한 어텐션 출력을 계산한다. 이는 근사(approximation)가 아닌 정확(exact) 연산이라는 점에서, 기존의 Sparse Attention이나 Linear Attention 근사 기법과 구별된다.

**성능 향상**: Flash Attention v2 (Dao, 2023)는 BERT-large 학습에서 표준 구현 대비 약 2.4배의 속도 향상을 보고하였으며, GPT-2 학습에서는 약 3배의 속도 향상을 달성하였다.

#### 3.8.2 Mixed Precision Training (Micikevicius et al., 2018)

Mixed Precision은 순전파(forward pass)와 역전파(backward pass)를 FP16 또는 BF16으로 수행하고, 마스터 가중치(master weights)는 FP32로 유지하는 기법이다.

```
학습 루프:
1. FP32 마스터 가중치 → FP16/BF16 사본 생성
2. 순전파/역전파를 FP16/BF16으로 수행
3. FP16/BF16 그래디언트를 FP32로 변환
4. FP32 마스터 가중치 업데이트
```

BF16(Brain Floating Point 16)은 FP16 대비 넓은 지수(exponent) 범위(8비트)를 가지므로, loss scaling 없이도 수치적으로 안정적이다. 현재 대부분의 LLM 학습에서는 BF16이 표준적으로 사용된다. 메모리 사용량은 FP32 대비 약 50% 감소하며, Tensor Core 활용으로 연산 속도도 향상된다.

#### 3.8.3 Gradient Checkpointing (Chen et al., 2016)

학습 시 역전파를 위해 순전파의 모든 중간 활성값(activation)을 저장하면 메모리 소비가 크다. Gradient Checkpointing은 일부 레이어의 활성값만 저장하고, 역전파 시 필요한 활성값을 재계산(recomputation)한다. 이를 통해 활성값 메모리를 O(L)에서 O(√L)로 감소시킬 수 있으나(L은 레이어 수), 약 33%의 추가 연산이 발생한다.

#### 3.8.4 Curriculum Learning

Curriculum Learning은 학습 데이터를 "쉬운 것에서 어려운 것으로" 순차적으로 제시하는 기법이다. LLM 사전 학습에서는 짧은 시퀀스로 학습을 시작하여 점진적으로 시퀀스 길이를 증가시키거나, 높은 품질의 데이터를 학습 후반부에 집중 배치하는 방식이 적용된다. LLaMA-3에서는 학습 후반부에 코드와 수학 데이터의 비율을 증가시키는 데이터 혼합 스케줄(annealing)을 사용하였다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 표현 학습의 진화 (2013-2018)

```
Word2Vec (2013): 정적 단어 임베딩
  │ 문제: 다의어 처리 불가, 문맥 무시
  ▼
GloVe (2014): 전역 동시출현 통계 기반 임베딩
  │ 문제: 여전히 정적, 문맥 독립적
  ▼
ELMo (2018): BiLSTM 기반 문맥 의존 표현
  │ 문제: LSTM의 순차적 연산 → 병렬화/확장 한계
  ▼
Transformer (2017): Self-Attention으로 병렬화 달성
  │ 해결: 임의 길이 시퀀스를 O(1) 깊이로 처리
  │ 문제: O(n^2) 메모리 복잡도
  ▼
BERT (2018): Transformer Encoder + MLM 사전 학습
  │ 해결: 양방향 문맥 활용, NLU SOTA 달성
  │ 문제: 텍스트 생성 불가 (Encoder-only 한계)
```

### 4.2 생성 모델로의 전환 (2019-2022)

```
GPT-2 (2019): Decoder-only CLM, zero-shot 능력 시연
  │ 문제: 1.5B 규모, 성능 불충분
  ▼
GPT-3 (2020): 175B로 규모 확장, few-shot ICL 발견
  │ 해결: 규모 확장으로 창발적 능력 획득
  │ 문제: 300B 토큰만 학습 (under-trained)
  ▼
T5 (2020): Encoder-Decoder, 체계적 사전 학습 비교 연구
  │ 문제: 추론 효율에서 Decoder-only 대비 불리
  ▼
Kaplan Scaling Laws (2020): "모델 크기 우선 확장" 법칙
  │ 문제: 실험 설계 결함 (데이터 크기 과소평가)
  ▼
Chinchilla (2022): "데이터와 모델을 동등하게 확장" 법칙
  │ 해결: 최적 학습 자원 배분 원칙 정립
  │ 문제: 추론 비용은 모델 크기에 비례 → 배포 비효율
```

### 4.3 오픈소스 확산과 효율화 (2023-현재)

```
LLaMA-1 (2023.02): 공개 데이터, Chinchilla-optimal, 오픈소스
  │ 해결: 독점 모델에 필적하는 공개 모델
  │ 문제: 연구 전용 라이선스, 상업 활용 불가
  ▼
LLaMA-2 (2023.07): 상업 라이선스, 2T 토큰, GQA 도입
  │ 해결: 오픈소스 LLM 상업 생태계 촉발
  │ 문제: GPT-4 대비 여전히 성능 격차
  ▼
Mistral 7B (2023.09): SWA + GQA, 7B로 13B급 성능
  │ 해결: 소규모 모델의 효율적 아키텍처
  ▼
Mixtral 8x7B (2024.01): MoE로 파라미터 효율 극대화
  │ 해결: 활성 파라미터 12.9B로 70B급 성능
  ▼
LLaMA-3 (2024.04): 15T+ 토큰, 극단적 over-training
  │ 해결: 추론 효율 극대화 (작은 모델 + 대량 데이터)
  │ 문제: 학습 비용 급증 (30.84M GPU-hours)
  ▼
데이터 품질 연구 (FineWeb, Dolma, 2024):
  │ 해결: 공개 데이터 파이프라인의 체계화
  │ 현재 과제: 합성 데이터, 데이터 저작권, 다국어 확장
```

### 4.4 분산 학습 기법의 진화

```
Data Parallelism: 모델 복제 → GPU 메모리 한계
  ▼
ZeRO (2020): 모델 상태 분산 → 메모리 효율 해결
  ▼
Megatron-LM TP (2019): 레이어 내부 분할 → 노드 내 병렬화
  ▼
GPipe PP (2019): 레이어 그룹 분할 → 노드 간 병렬화
  ▼
3D Parallelism (2021): DP + TP + PP 통합 → 만 단위 GPU 활용
  ▼
FSDP (PyTorch): ZeRO의 프레임워크 표준화
  ▼
Flash Attention (2022): IO-aware 어텐션 → 학습/추론 속도 향상
```

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회/출처 | 링크 |
|---|------|------|-----------|------|
| 1 | Efficient Estimation of Word Representations in Vector Space (Word2Vec) | Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean | **ICLR 2013 Workshop** | https://arxiv.org/abs/1301.3781 |
| 2 | GloVe: Global Vectors for Word Representation | Jeffrey Pennington, Richard Socher, Christopher D. Manning | **EMNLP 2014** | https://nlp.stanford.edu/pubs/glove.pdf |
| 3 | Deep contextualized word representations (ELMo) | Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer | **NAACL 2018** | https://arxiv.org/abs/1802.05365 |
| 4 | Attention Is All You Need | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin | **NeurIPS 2017** | https://arxiv.org/abs/1706.03762 |
| 5 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova | **NAACL 2019** | https://arxiv.org/abs/1810.04805 |
| 6 | Language Models are Unsupervised Multitask Learners (GPT-2) | Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever | OpenAI Technical Report, 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| 7 | Language Models are Few-Shot Learners (GPT-3) | Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Siber, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei | **NeurIPS 2020** | https://arxiv.org/abs/2005.14165 |
| 8 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu | **JMLR 2020** | https://arxiv.org/abs/1910.10683 |
| 9 | Scaling Laws for Neural Language Models | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei | arXiv 2020 | https://arxiv.org/abs/2001.08361 |
| 10 | Training Compute-Optimal Large Language Models (Chinchilla) | Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre | **NeurIPS 2022** | https://arxiv.org/abs/2203.15556 |
| 11 | LLaMA: Open and Efficient Foundation Language Models | Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample | arXiv 2023 | https://arxiv.org/abs/2302.13971 |
| 12 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom | arXiv 2023 | https://arxiv.org/abs/2307.09288 |
| 13 | The Llama 3 Herd of Models | Meta AI | arXiv 2024 | https://arxiv.org/abs/2407.21783 |
| 14 | Mistral 7B | Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothee Lacroix, William El Sayed | arXiv 2023 | https://arxiv.org/abs/2310.06825 |
| 15 | Mixtral of Experts | Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, Thibaut Lavril, Thomas Wang, Timothee Lacroix, William El Sayed | arXiv 2024 | https://arxiv.org/abs/2401.04088 |
| 16 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro | arXiv 2019 | https://arxiv.org/abs/1909.08053 |
| 17 | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen | **NeurIPS 2019** | https://arxiv.org/abs/1811.06965 |
| 18 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He | **SC 2020** | https://arxiv.org/abs/1910.02054 |
| 19 | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (3D Parallelism) | Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia | **SC 2021** | https://arxiv.org/abs/2104.04473 |
| 20 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re | **NeurIPS 2022** | https://arxiv.org/abs/2205.14135 |
| 21 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Tri Dao | **ICLR 2024** | https://arxiv.org/abs/2307.08691 |
| 22 | Mixed Precision Training | Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu | **ICLR 2018** | https://arxiv.org/abs/1710.03740 |
| 23 | Training Deep Nets with Sublinear Memory Cost (Gradient Checkpointing) | Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin | arXiv 2016 | https://arxiv.org/abs/1604.06174 |
| 24 | FineWeb: decanting the web for the finest text data at scale | Guilherme Penedo, Hynek Kydlicek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf | **NeurIPS 2024 Datasets and Benchmarks** | https://arxiv.org/abs/2406.17557 |
| 25 | Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research | Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, Kyle Lo | **ACL 2024** | https://arxiv.org/abs/2402.00159 |
| 26 | RedPajama: an Open Dataset for Training Large Language Models | Together Computer | 2023 | https://github.com/togethercomputer/RedPajama-Data |
| 27 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling | Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy | arXiv 2020 | https://arxiv.org/abs/2101.00027 |
| 28 | Deduplicating Training Data Makes Language Models Better | Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini | **ACL 2022** | https://arxiv.org/abs/2107.06499 |
| 29 | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu | **NeurIPS 2023** | https://arxiv.org/abs/2305.10429 |
| 30 | Scaling Data-Constrained Language Models | Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin Raffel | **NeurIPS 2023** | https://arxiv.org/abs/2305.16264 |
| 31 | PaLM: Scaling Language Modeling with Pathways | Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel | **JMLR 2023** | https://arxiv.org/abs/2204.02311 |
| 32 | OLMo: Accelerating the Science of Language Models | Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, Hannaneh Hajishirzi | **ACL 2024** | https://arxiv.org/abs/2402.00838 |
| 33 | RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE) | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu | **Neurocomputing 2024** | https://arxiv.org/abs/2104.09864 |
| 34 | GLU Variants Improve Transformer (SwiGLU) | Noam Shazeer | arXiv 2020 | https://arxiv.org/abs/2002.05202 |
| 35 | Root Mean Square Layer Normalization (RMSNorm) | Biao Zhang, Rico Sennrich | **NeurIPS 2019** | https://arxiv.org/abs/1910.07467 |
| 36 | Curriculum Learning | Yoshua Bengio, Jerome Louradour, Ronan Collobert, Jason Weston | **ICML 2009** | https://dl.acm.org/doi/10.1145/1553374.1553380 |
| 37 | Data-Efficient Language Models: Survey and Outlook | Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, Roy Ka-Wei Lee | arXiv 2024 | https://arxiv.org/abs/2402.13473 |
| 38 | Scaling Language Models: Methods, Analysis & Insights from Training Gopher | Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terber, Cristian Voicu, Jost Tobias Springenberg, Razvan Pascanu, Siamak Shakeri, Koray Kavukcuoglu, Lila Fontes, Chris Olah, Wenhao Li, Michael Sherburn, Nando de Freitas, Oriol Vinyals | arXiv 2021 | https://arxiv.org/abs/2112.11446 |
