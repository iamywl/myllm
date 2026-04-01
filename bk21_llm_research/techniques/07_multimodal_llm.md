# Multimodal LLM (멀티모달 대규모 언어 모델)

## 1. 기법의 정의

Multimodal Large Language Model(MLLM)은 텍스트 이외의 모달리티(이미지, 비디오, 오디오 등)를 입력으로 수용하여 대규모 언어 모델(LLM)의 추론 및 생성 능력을 다중 감각 정보에 확장하는 기술이다. 구체적으로, 사전 학습된 비전 인코더(vision encoder)가 추출한 시각적 특징을 프로젝션 레이어(projection layer)를 통해 LLM의 임베딩 공간으로 매핑한 후, 텍스트 토큰과 연결(concatenation)하여 자기회귀적(autoregressive) 생성을 수행하는 아키텍처가 핵심이다.

MLLM의 일반적 아키텍처는 다음 세 가지 구성 요소로 정의된다:

```
[이미지/비디오/오디오] → [모달리티 인코더] → 특징 벡터 시퀀스
                                                ↓
                                        [프로젝션 레이어]
                                                ↓
[텍스트 프롬프트] → [Tokenizer] → 텍스트 토큰 → [LLM Backbone] → 응답 생성
```

| 구성 요소 | 역할 | 대표적 선택지 |
|-----------|------|--------------|
| **모달리티 인코더** | 비텍스트 입력을 고차원 특징으로 변환 | CLIP ViT-L/14, SigLIP, InternViT-6B, EVA-CLIP |
| **프로젝션 레이어** | 인코더 특징을 LLM 임베딩 공간에 정렬 | Linear, 2-layer MLP, Q-Former, Perceiver Resampler |
| **LLM 백본** | 멀티모달 토큰 시퀀스에 대한 추론 및 텍스트 생성 | LLaMA-2/3, Vicuna, InternLM2, Qwen-2 |

이 정의에서 핵심적 기술적 질문은 세 가지이다: (1) 어떤 비전 인코더를 선택할 것인가, (2) 프로젝션 레이어의 구조적 복잡도를 어떻게 설정할 것인가, (3) 학습 단계(training stage)를 어떻게 설계하여 모달리티 간 정렬(alignment)과 지시 수행(instruction following) 능력을 동시에 확보할 것인가이다.

---

## 2. 기존 기법의 한계와 MLLM 등장 배경

### 2.1 텍스트 전용 LLM의 근본적 한계

GPT-3(Brown et al., 2020), LLaMA(Touvron et al., 2023) 등 텍스트 전용 LLM은 자연어 추론에서 인간 수준의 성능을 달성하였으나, 현실 세계 정보의 상당 부분을 차지하는 시각/청각 정보를 전혀 처리할 수 없다는 구조적 제약이 존재하였다. "이 X-ray 이미지에서 이상 소견이 있는가?" 또는 "이 차트의 추세를 분석하라"와 같은 질문에 텍스트 전용 모델은 원천적으로 응답이 불가능하였다.

### 2.2 독립 비전-언어 모델의 한계

MLLM 이전의 비전-언어(Vision-Language) 연구는 태스크별 독립 모델 패러다임에 의존하였다. VQA(Visual Question Answering)를 위한 모델, 이미지 캡셔닝을 위한 모델, 시각적 추론을 위한 모델이 각각 별도로 학습되었으며, 이들 간의 지식 공유가 불가능하였다.

**문제점:**
- 태스크별 별도의 헤드(head) 설계 및 학습이 필요하여 확장성이 낮았다
- 자유형(open-ended) 응답 생성이 불가능하고 분류/선택형 출력에 국한되었다
- 복합적 시각 추론(예: "이 두 이미지의 차이를 설명하고 원인을 추론하라")이 불가능하였다

### 2.3 CLIP의 돌파구와 새로운 문제

Radford et al.(2021)의 CLIP은 4억 개의 이미지-텍스트 쌍에 대해 대조 학습(contrastive learning)을 수행하여, 이미지와 텍스트를 동일한 임베딩 공간에 정렬하는 데 성공하였다. 이는 zero-shot 이미지 분류에서 기존 지도학습 모델에 필적하는 성능을 보여주었다.

**CLIP이 해결한 문제:** 이미지-텍스트 간 의미적 정렬(semantic alignment)을 범용적으로 학습할 수 있음을 증명하였다.

**CLIP이 남긴 문제:** 대조 학습은 본질적으로 판별(discriminative) 태스크에 최적화되어 있어, 자유형 텍스트 생성이 불가능하였다. 즉, "이 이미지가 고양이인가?"에는 답할 수 있지만, "이 이미지를 자세히 설명하라"에는 답할 수 없었다.

이 간극을 해소하기 위해, 사전 학습된 비전 인코더(주로 CLIP ViT)의 특징을 LLM의 생성 능력과 결합하는 MLLM 패러다임이 등장하였다.

---

## 3. 주요 기법 상세

### 3.1 CLIP (Contrastive Language-Image Pre-training)

**문제:** 기존 이미지 분류는 고정된 클래스 레이블에 의존하여, 새로운 카테고리에 대한 일반화가 불가능하였다.

**해결:** 이미지 인코더(ViT)와 텍스트 인코더(Transformer)를 쌍으로 학습하여, 이미지-텍스트 임베딩의 코사인 유사도를 최대화/최소화하는 대조 학습을 수행하였다. WebImageText(WIT) 데이터셋 4억 쌍을 활용하였다.

```
손실 함수: InfoNCE Loss
L = -log( exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ) )
```

**핵심 기여:** CLIP ViT-L/14(@336px)는 이후 거의 모든 MLLM의 기본 비전 인코더로 채택되었다. 이는 CLIP의 시각적 표현이 언어와의 정렬에 적합함을 의미한다.

**남은 문제:** 판별 모델이므로 생성 능력이 부재하였으며, 세밀한 공간적 이해(fine-grained spatial understanding)가 부족하였다.

### 3.2 Flamingo

**문제:** CLIP은 생성이 불가능하고, 기존 VLP(Vision-Language Pre-training) 모델은 few-shot 적응 능력이 부족하였다.

**해결:** Alayrac et al.(2022)은 사전 학습된 비전 인코더(NFNet)와 LLM(Chinchilla)을 모두 동결(freeze)한 상태에서, Perceiver Resampler와 Gated Cross-Attention 레이어만을 학습하는 아키텍처를 제안하였다.

**Perceiver Resampler의 구조:**
- 가변 길이 이미지 특징을 고정 개수(64개)의 시각 토큰으로 압축한다
- 학습 가능한 쿼리 벡터가 이미지 특징에 대해 cross-attention을 수행한다
- 이 구조로 이미지 해상도와 무관하게 일정한 수의 토큰을 LLM에 공급한다

**Gated Cross-Attention:**
- LLM의 기존 self-attention 레이어 사이에 cross-attention 레이어를 삽입한다
- 게이트 파라미터(tanh gate)를 0으로 초기화하여, 학습 초기에는 원래 LLM 동작을 보존한다

**벤치마크 성과:** 32-shot 세팅에서 OKVQA 57.8%, VQAv2 67.6%를 기록하였다.

**남은 문제:** 모델이 비공개(closed-source)이며, Gated Cross-Attention 구조가 LLM 아키텍처에 대한 수정을 요구하여 범용성이 낮았다.

### 3.3 BLIP-2

**문제:** Flamingo의 Gated Cross-Attention은 LLM 내부 구조를 변경해야 하며, 대규모 이미지-텍스트 데이터로 전체 모델을 학습하는 것은 연산 비용이 과도하였다.

**해결:** Li et al.(2023)은 Q-Former(Querying Transformer)라는 경량 브릿지 모듈을 제안하여, 동결된 비전 인코더와 동결된 LLM 사이를 연결하였다.

**Q-Former 아키텍처 상세:**
- 32개의 학습 가능한 쿼리 토큰이 비전 인코더 출력에 대해 cross-attention을 수행한다
- Self-attention은 쿼리 토큰과 텍스트 토큰 모두에 적용된다
- 3가지 사전 학습 목표: ITC(Image-Text Contrastive), ITG(Image-grounded Text Generation), ITM(Image-Text Matching)

**2단계 학습:**
1. **Stage 1 (Vision-Language Representation Learning):** Q-Former를 비전 인코더와 함께 학습하여 시각-언어 정렬을 수행한다. 129M 이미지-텍스트 쌍을 사용한다.
2. **Stage 2 (Vision-to-Language Generative Learning):** Q-Former 출력을 FC 레이어로 투사하여 동결된 LLM(FlanT5-XXL 또는 OPT-6.7B)에 입력한다.

**벤치마크 성과:** VQAv2에서 FlanT5-XXL 기반으로 82.6%를 달성하였으며, 학습 파라미터는 188M에 불과하였다.

**남은 문제:** Q-Former의 32개 토큰으로 압축하는 과정에서 세밀한 시각 정보가 손실되었다. 또한 Q-Former의 3단계 사전 학습 과정이 복잡하여 재현성이 낮았다.

### 3.4 LLaVA (Large Language and Vision Assistant)

**문제:** BLIP-2의 Q-Former는 구조가 복잡하고 학습 과정이 다단계이다. 또한, 기존 MLLM은 시각적 지시 수행(visual instruction following) 능력이 부족하였다.

**해결:** Liu et al.(2023)은 두 가지 핵심적 단순화를 제안하였다:

**(1) 단순한 프로젝션:** CLIP ViT-L/14 출력을 단일 선형 레이어(linear projection)로 LLM 입력 공간에 매핑한다. Q-Former 대비 극도로 간단하지만, 적절한 데이터와 학습 전략이 결합되면 충분히 효과적임을 증명하였다.

**(2) Visual Instruction Tuning 데이터 생성:** GPT-4를 활용하여 COCO 이미지의 캡션과 바운딩 박스 정보를 기반으로 158K개의 멀티턴 시각 지시문(visual instruction-following data)을 자동 생성하였다. 데이터 유형은 세 가지이다:
- **대화(Conversation):** 이미지에 대한 다중 턴 질의응답
- **상세 설명(Detailed Description):** 이미지의 포괄적 서술
- **복합 추론(Complex Reasoning):** 다단계 논리적 추론이 필요한 질문

**학습 과정:**
| 단계 | 데이터 | 학습 대상 | 에폭 |
|------|--------|-----------|------|
| Pre-training | 595K CC3M 필터링 캡션 | Linear projection만 | 1 |
| Instruction Tuning | 158K GPT-4 생성 데이터 | Projection + LLM 전체 | 3 |

**남은 문제:** 단일 선형 레이어의 표현력 한계, 336px 고정 해상도로 인한 세밀한 시각 인식 부족, 학습 데이터 규모가 158K로 제한적이었다.

### 3.5 LLaVA-1.5 / LLaVA-NeXT

**LLaVA-1.5가 해결한 문제:**

Liu et al.(2023b)은 LLaVA의 한계를 체계적으로 분석하여 최소한의 수정으로 대폭적 성능 향상을 달성하였다.

**핵심 개선 사항:**

| 변경 사항 | LLaVA | LLaVA-1.5 | 효과 |
|-----------|-------|-----------|------|
| 프로젝션 레이어 | Linear (1-layer) | MLP (2-layer, GELU) | +3~5% 벤치마크 향상 |
| 이미지 해상도 | 224px → 336px | 336px | 세부 인식 향상 |
| 학습 데이터 | 158K | 665K (ShareGPT 포함) | 지시 수행 능력 대폭 향상 |
| LLM 백본 | Vicuna-7B/13B | Vicuna-7B/13B | 동일 |
| Academic 데이터 | 미포함 | VQA, OCR, GQA 등 추가 | 학술 벤치마크 성능 향상 |

**벤치마크 성과 (LLaVA-1.5-13B):**

| 벤치마크 | 점수 |
|----------|------|
| MMBench | 67.7 |
| SEED-Bench | 68.2 |
| MM-Vet | 35.4 |
| POPE (hallucination) | 85.9 |

**LLaVA-NeXT가 해결한 문제:** 고정 해상도의 근본적 한계를 극복하기 위해 동적 고해상도(dynamic high-resolution) 전략을 도입하였다.

**AnyRes 전략:**
- 입력 이미지를 최적 그리드(예: 2×2, 1×3, 3×1 등)로 분할한다
- 각 패치를 독립적으로 CLIP ViT에 통과시킨다
- 전체 이미지의 축소판(thumbnail)도 별도로 인코딩하여 글로벌 컨텍스트를 보존한다
- 패치별 특징과 글로벌 특징을 연결(concatenate)하여 LLM에 입력한다

이 전략으로 672px, 1008px 등 고해상도 이미지 처리가 가능해졌으나, 시각 토큰 수가 수천 개로 증가하여 추론 지연(inference latency)과 컨텍스트 소모가 새로운 문제로 대두되었다.

**LLaVA-NeXT 벤치마크 성과:**

| 벤치마크 | LLaVA-1.5-13B | LLaVA-NeXT-13B | 향상 |
|----------|--------------|----------------|------|
| MMBench | 67.7 | 70.0 | +2.3 |
| TextVQA | 61.3 | 67.1 | +5.8 |
| DocVQA | - | 74.4 | - |

### 3.6 InternVL / InternVL-1.5 / InternVL-2

**문제:** 기존 MLLM은 CLIP ViT-L/14(304M 파라미터)에 의존하여 비전 인코더의 표현력에 상한이 존재하였다. 또한 CLIP은 자연 이미지에 편향되어 문서, 차트, 수식 등 학술적 시각 자료에 대한 이해가 부족하였다.

**해결:** Chen et al.(2024)은 InternViT-6B라는 60억 파라미터 규모의 비전 인코더를 자체 학습하여, 비전 인코더 자체의 성능 상한을 끌어올렸다.

**InternVL 아키텍처:**
- **비전 인코더:** InternViT-6B (ViT-Huge 대비 20배 대형)
- **LLM 백본:** InternLM2-Chat (7B/20B)
- **프로젝션:** Dynamic resolution + pixel shuffle로 토큰 수를 1/4로 압축
- **학습 데이터:** 대규모 다국어 멀티모달 데이터 (중국어 포함)

**InternVL-1.5의 핵심 혁신 — Dynamic Resolution:**
- 이미지를 448×448 타일로 분할하되, 최대 12개 타일까지 허용한다
- Pixel shuffle 연산으로 각 타일의 토큰 수를 256개에서 64개로 압축한다
- 이로써 고해상도 이미지를 효율적으로 처리한다

**InternVL-2 벤치마크 성과 (InternVL2-Llama3-76B):**

| 벤치마크 | 점수 | 비교 (GPT-4o) |
|----------|------|---------------|
| MMBench-EN | 86.5 | 83.4 |
| MMMU | 55.2 | 69.1 |
| MathVista | 65.6 | 63.8 |
| DocVQA | 94.1 | 92.8 |
| ChartQA | 84.9 | 85.7 |

InternVL2-76B는 다수의 벤치마크에서 GPT-4o에 필적하거나 초과하는 성능을 보여, 오픈소스 MLLM이 상용 모델과의 격차를 크게 줄였음을 입증하였다.

**남은 문제:** 76B 규모의 모델은 배포 비용이 높으며, 비전 인코더 6B + LLM 70B의 조합은 단일 GPU 추론이 불가능하다.

### 3.7 Qwen-VL / Qwen2-VL

**문제:** 기존 MLLM은 주로 영어 데이터로 학습되어 다국어 시각-언어 이해 능력이 제한적이었으며, 바운딩 박스 기반 시각적 그라운딩(visual grounding) 능력이 부족하였다.

**해결:** Bai et al.(2023)은 Qwen-VL에서 다국어(중국어, 영어, 다국어) 지원과 시각적 그라운딩을 핵심 설계 목표로 설정하였다.

**Qwen-VL 아키텍처:**
- **비전 인코더:** ViT-bigG/14 (OpenCLIP, 1.9B 파라미터)를 448×448 해상도로 사용
- **프로젝션:** 단일 cross-attention 레이어로 256개 이미지 토큰을 64개로 압축
- **LLM:** Qwen-7B
- **특수 토큰:** `<img>`, `</img>`, `<ref>`, `</ref>`, `<box>`, `</box>` 등을 도입하여 이미지 위치와 바운딩 박스를 텍스트 시퀀스 내에서 표현

**3단계 학습:**
1. **Stage 1:** 1.4B 이미지-텍스트 쌍으로 비전 인코더 + 프로젝션 학습 (LLM 동결)
2. **Stage 2:** 고품질 멀티태스크 데이터로 전체 모델 학습 (VQA, 캡셔닝, 그라운딩 등 7개 태스크)
3. **Stage 3:** 지시 수행 데이터로 Supervised Fine-tuning

**Qwen2-VL(2024)의 주요 개선:**
- Naive Dynamic Resolution: 이미지를 리사이즈 없이 가변 해상도 그대로 처리한다
- M-RoPE(Multimodal Rotary Position Embedding): 시간, 높이, 너비 3차원의 위치 정보를 RoPE에 통합하여 비디오의 시공간적 위치를 인코딩한다
- 비전 인코더를 675M ViT로 교체하고 학습 시 동결하지 않는다(unfrozen)

**벤치마크 성과 (Qwen2-VL-72B):**

| 벤치마크 | 점수 |
|----------|------|
| MMMU | 64.5 |
| MathVista | 70.5 |
| DocVQA | 96.5 |
| RealWorldQA | 77.8 |

### 3.8 MoAI / CoLLaVO

**문제:** 단일 비전 인코더(예: CLIP ViT)만으로는 객체 검출, 깊이 추정, OCR 등 다양한 시각적 세부 정보를 충분히 포착할 수 없었다.

**MoAI (Mixture of All Intelligence)의 해결:**

Lee et al.(2024)은 다수의 전문화된 컴퓨터 비전(CV) 모델의 출력을 MLLM의 입력으로 통합하는 MoAI 아키텍처를 제안하였다.

**아키텍처 상세:**
- **보조 CV 모델:** Open-vocabulary 객체 검출기(OWL-ViT), 세그멘테이션 모델(SAM), OCR 엔진, 깊이 추정 모델(DPT)의 출력을 수집한다
- **MoAI-Compressor:** 보조 모델들의 출력을 압축하여 고정 길이의 보조 토큰으로 변환한다
- **MoAI-Mixer:** Mixture of Experts 라우터가 원본 비전 토큰과 보조 토큰을 동적으로 혼합한다

**벤치마크 성과 (MoAI-7B):**

| 벤치마크 | MoAI-7B | LLaVA-1.5-13B |
|----------|---------|---------------|
| MMBench | 77.2 | 67.7 |
| MM-Vet | 43.7 | 35.4 |
| POPE | 87.1 | 85.9 |
| AI2D | 78.1 | 61.1 |

7B 규모에서 13B 모델을 압도하는 성능을 달성하여, 보조 CV 모델의 정보가 MLLM 성능에 결정적임을 입증하였다.

**CoLLaVO (Crayon Large Language and Vision mOdel):**

Lee et al.(2024b)은 객체 수준의 시각적 프롬프팅(visual prompting)을 통해 MLLM의 객체 인식 능력을 향상시키는 CoLLaVO를 제안하였다. Crayon Prompt라는 기법으로 이미지 내 객체에 색상 마커를 오버레이하고, 이를 통해 LLM이 개별 객체를 명시적으로 인식하도록 유도하였다.

**남은 문제:** 보조 CV 모델의 추론 비용이 추가되며, 보조 모델의 오류가 전파(error propagation)될 수 있다.

### 3.9 GPT-4V / GPT-4o

**GPT-4V(ision):**

OpenAI(2023)가 공개한 GPT-4V는 최초의 상용 수준 MLLM로, 이미지 이해, OCR, 차트 해석, 공간 추론, 유머 이해 등 광범위한 시각-언어 태스크에서 당시 SOTA 성능을 기록하였다.

**알려진 아키텍처 정보:**
- 정확한 아키텍처는 비공개이나, 비전 인코더 + 대형 LLM 백본 구조로 추정된다
- 타일 기반 고해상도 처리를 지원한다
- RLHF를 통한 안전성 정렬이 적용되어 있다

**GPT-4o (omni):**

2024년 출시된 GPT-4o는 텍스트, 이미지, 오디오를 단일 모델에서 네이티브(native)하게 처리하는 진정한 옴니모달(omni-modal) 모델이다. 기존의 "인코더 + 프로젝션 + LLM" 파이프라인 구조가 아닌, 모든 모달리티를 단일 토큰 공간에서 통합 처리하는 것으로 알려져 있다.

**GPT-4o의 핵심 차별점:**
- 오디오 입출력을 별도 ASR/TTS 모듈 없이 end-to-end로 처리한다
- 이미지 생성 능력이 통합되어 있다 (2025년 업데이트)
- 응답 지연(latency)이 GPT-4V 대비 2배 이상 빠르다

**벤치마크 성과 (GPT-4o, 2024.05 기준):**

| 벤치마크 | GPT-4o | GPT-4V |
|----------|--------|--------|
| MMMU | 69.1 | 56.8 |
| MMBench-EN | 83.4 | 77.0 |
| MathVista | 63.8 | 58.2 |
| HallusionBench | 55.0 | 43.9 |

**남은 문제:** 완전 비공개 모델이므로 학술적 재현이 불가능하며, API 비용이 연구 목적으로는 과도하다.

### 3.10 Gemini

**문제:** 기존 MLLM은 대부분 "사전 학습된 인코더 + LLM" 조합이라는 모듈식(modular) 접근에 의존하여, 모달리티 간 심층적 통합(deep fusion)에 한계가 있었다.

**해결:** Google DeepMind의 Gemini(Team et al., 2023)는 처음부터 멀티모달 데이터로 사전 학습된 네이티브 멀티모달 모델이다. 텍스트, 이미지, 오디오, 비디오를 단일 모델에서 인코딩하고 생성할 수 있다.

**Gemini 계열:**
- **Gemini Ultra:** MMLU에서 인간 전문가 수준(90.0%) 최초 달성
- **Gemini Pro:** 중간 규모, API 제공
- **Gemini Nano:** 모바일 배포용 경량 모델

**Gemini 1.5 Pro의 핵심 혁신:**
- 100만 토큰 이상의 장문맥(long context) 지원
- 비디오 1시간 분량을 단일 컨텍스트에서 처리 가능
- Mixture of Experts 아키텍처로 효율성 확보

**벤치마크 성과 (Gemini Ultra):**

| 벤치마크 | Gemini Ultra |
|----------|-------------|
| MMMU | 59.4 |
| MathVista | 53.0 |
| MMLU | 90.0 |
| TextVQA | 82.3 |

**남은 문제:** 비공개 모델이며, 환각(hallucination) 현상이 여전히 보고된다.

### 3.11 Video-LLM

**문제:** 이미지 기반 MLLM을 비디오에 직접 적용하면 프레임 수에 비례하여 시각 토큰이 폭발적으로 증가한다. 예를 들어, 30fps 1분 비디오는 1800프레임이며, 프레임당 576 토큰이면 총 103만 토큰이 되어 현실적으로 처리가 불가능하다.

**주요 접근법:**

**(1) Video-LLaVA (Lin et al., 2023):**
- 이미지와 비디오를 통합된 시각 표현(unified visual representation)으로 학습한다
- LanguageBind 인코더를 사용하여 이미지/비디오 모두 동일한 특징 공간으로 매핑한다
- 핵심 기여: "alignment before projection" — 프로젝션 이전에 이미지-비디오 표현을 정렬하여 상호 학습 효과를 극대화한다

**(2) LLaVA-Video (Zhang et al., 2024):**
- SlowFast 전략으로 비디오 프레임을 효율적으로 샘플링한다
- Slow pathway: 소수 프레임을 고해상도로 처리 (시각적 세부 정보)
- Fast pathway: 다수 프레임을 저해상도로 처리 (시간적 동태)

**(3) VideoChat2 (Li et al., 2024):**
- 프로그레시브 학습 전략으로 이미지→비디오→지시 수행 순서로 점진적 학습한다

**비디오 토큰 압축 전략:**

| 전략 | 방법 | 압축률 |
|------|------|--------|
| 균일 샘플링 | N개 프레임만 추출 | 높음 (정보 손실 대) |
| 시간적 풀링 | 인접 프레임 특징 평균화 | 중간 |
| Q-Former 기반 | 쿼리로 비디오 특징 압축 | 높음 (학습 필요) |
| SlowFast | 이중 해상도 샘플링 | 중간 |
| Token Merging | 유사 토큰 병합 | 적응적 |

**남은 문제:** 장시간(10분 이상) 비디오의 시간적 추론, 비디오 내 인과적 사건 이해, 실시간 비디오 스트림 처리가 미해결이다.

### 3.12 Audio-LLM

**문제:** 시각적 멀티모달 연구에 비해 오디오-언어 통합은 상대적으로 덜 탐구되었다. 기존 음성 인식(ASR)은 전사(transcription)에 국한되어 있으며, 음성의 감정, 화자 특성, 환경 소리 등 비언어적 오디오 정보의 이해와 추론이 불가능하였다.

**주요 모델:**

**(1) SALMONN (Tang et al., 2023):**
- 이중 오디오 인코더를 사용한다: Whisper(음성) + BEATs(오디오 이벤트)
- Q-Former로 오디오 특징을 압축하여 LLM(Vicuna)에 입력한다
- 음성 인식, 오디오 캡셔닝, 음악 분석을 단일 모델에서 수행한다

**(2) Qwen-Audio (Chu et al., 2023):**
- Whisper-large-v2를 오디오 인코더로 사용한다
- 30개 이상의 오디오 태스크를 멀티태스크 학습한다
- 태스크 식별을 위한 특수 토큰 없이 자연어 지시만으로 태스크를 구분한다

**(3) Qwen2-Audio (Chu et al., 2024):**
- 음성 이해와 오디오 분석을 통합하고, 음성 기반 대화(voice chat)를 지원한다

**남은 문제:** 오디오-비전-텍스트 3개 모달리티의 동시 처리, 실시간 음성 대화에서의 지연 시간 최소화가 핵심 과제이다.

---

## 4. 기법 진화의 인과적 흐름

### 4.1 프로젝션 레이어 설계의 진화

MLLM 발전의 핵심 축 중 하나는 비전 인코더와 LLM 사이의 프로젝션 레이어 설계이다. 이 진화는 다음과 같은 인과적 흐름을 따른다:

```
Perceiver Resampler (Flamingo, 2022)
  → 가변 길이 → 고정 길이 압축, 그러나 LLM 내부 수정 필요
  → 문제: 아키텍처 종속성
Q-Former (BLIP-2, 2023)
  → 독립적 브릿지 모듈, 양쪽 모두 동결 가능
  → 문제: 3단계 학습 복잡, 정보 손실
Linear Projection (LLaVA, 2023)
  → 극도의 단순함, 비전 토큰을 직접 LLM에 주입
  → 문제: 표현력 부족
2-layer MLP (LLaVA-1.5, 2023)
  → Linear 대비 +3~5% 성능, 여전히 간단
  → 문제: 토큰 수 압축 불가
Pixel Shuffle + MLP (InternVL-1.5, 2024)
  → 공간적 다운샘플링으로 토큰 수 1/4 압축
  → 문제: 정보 손실과 효율의 트레이드오프
```

**핵심 교훈:** 프로젝션 레이어의 복잡도보다 학습 데이터의 품질과 양이 성능에 더 큰 영향을 미친다는 것이 LLaVA 계열의 실험으로 반복 검증되었다. 이는 "데이터 중심(data-centric)" 접근의 중요성을 시사한다.

### 4.2 비전 인코더 선택의 진화

```
CLIP ViT-L/14 (304M, 224px/336px)
  → 대부분의 초기 MLLM이 채택, 검증된 시각-언어 정렬
  → 문제: 해상도 제한, 파라미터 규모 한계
SigLIP (Google, 2023)
  → Sigmoid loss로 배치 크기 제약 해소, 더 효율적 학습
  → PaLI-X, PaLI-3에서 채택
EVA-CLIP ViT-E (4.4B)
  → CLIP 대비 대형화, 향상된 시각 표현
  → 문제: 추론 비용 증가
InternViT-6B (InternVL, 2024)
  → 자체 학습한 초대형 비전 인코더, MLLM 특화 학습
  → 문제: 학습 비용 과도
다중 인코더 앙상블 (Cambrian-1, 2024)
  → CLIP + DINOv2 + SigLIP 등 복수 인코더 결합
  → 문제: 추론 비용 선형 증가
```

Tong et al.(2024)의 Cambrian-1은 다양한 비전 인코더의 조합을 체계적으로 실험한 결과, 단일 인코더보다 이질적(heterogeneous) 인코더의 조합이 일관되게 우수함을 보고하였다. 특히 CLIP(언어 정렬)과 DINOv2(자기지도 시각 학습)의 조합이 상보적 효과를 발휘하였다.

### 4.3 학습 데이터 구축의 진화

Visual Instruction Tuning 데이터의 구축 방법론은 MLLM 성능에 결정적 영향을 미치며, 다음과 같이 진화하였다:

```
GPT-4 기반 자동 생성 (LLaVA, 158K)
  → COCO 캡션 + bbox → GPT-4로 QA 생성
  → 문제: GPT-4 API 비용, 생성 품질 편차
ShareGPT4V (Chen et al., 2023)
  → GPT-4V로 직접 이미지를 보고 고품질 캡션 생성 (100K)
  → 문제: API 비용 더 증가
ALLaVA (Chen et al., 2024)
  → GPT-4V로 생성한 후 MLLM 자체로 bootstrapping
LVIS-Instruct-4V (Wang et al., 2024)
  → 객체 카테고리 다양성을 확보한 220K 지시문
자체 합성 (Self-synthesis)
  → 학습된 MLLM이 자체적으로 데이터를 생성/정제하는 반복적 접근
```

### 4.4 해상도 처리 전략의 진화

```
고정 저해상도 (224px/336px)
  → CLIP 기본 입력 크기 사용
  → 문제: OCR, 문서, 원거리 객체 인식 불가
고정 고해상도 (448px/672px)
  → 비전 인코더를 높은 해상도로 fine-tune
  → 문제: 연산량 해상도^2에 비례 증가
타일 기반 동적 해상도 (LLaVA-NeXT, InternVL-1.5)
  → 이미지를 그리드로 분할하여 타일별 인코딩
  → 문제: 토큰 수 가변, 타일 경계에서 정보 단절
네이티브 가변 해상도 (Qwen2-VL)
  → 리사이즈 없이 원본 비율 유지, M-RoPE로 위치 인코딩
  → 현재까지 가장 유연한 해상도 처리
```

### 4.5 모달리티 확장의 진화 흐름

```
이미지 전용 (CLIP → LLaVA → InternVL)
  ↓
이미지 + 비디오 (Video-LLaVA, LLaVA-Video)
  ↓
이미지 + 오디오 (SALMONN, Qwen-Audio)
  ↓
이미지 + 비디오 + 오디오 (GPT-4o, Gemini)
  ↓
옴니모달 입출력 (GPT-4o: 이미지 생성 포함)
```

이 흐름에서 핵심적 전환점은 "모듈식 조합(modular composition)"에서 "네이티브 통합(native integration)"으로의 이동이다. 초기 모델은 사전 학습된 인코더를 결합하는 방식이었으나, GPT-4o와 Gemini는 처음부터 멀티모달 데이터로 사전 학습하여 모달리티 간 심층적 상호작용을 가능하게 하였다.

### 4.6 현재의 핵심 미해결 문제

**(1) 멀티모달 환각(Multimodal Hallucination):**
MLLM은 이미지에 존재하지 않는 객체나 속성을 생성하는 환각 현상이 심각하다. POPE 벤치마크에서 최선의 모델도 약 85~90% 정확도에 머물러 있다. Li et al.(2023)의 분석에 따르면, 이는 LLM의 언어적 편향(language prior)이 시각 정보를 압도하는 현상에 기인한다.

**(2) 시각 토큰 효율성:**
고해상도 이미지나 비디오 처리 시 수천~수만 개의 시각 토큰이 생성되어 LLM의 컨텍스트 윈도우를 소모한다. 토큰 압축, 토큰 병합(token merging), 적응적 샘플링 등의 연구가 활발히 진행 중이다.

**(3) 세밀한 공간 추론(Fine-grained Spatial Reasoning):**
"왼쪽에서 세 번째 객체의 색상은?" 같은 세밀한 공간적 질문에 대한 정확도가 여전히 낮다. 이는 비전 인코더의 공간 정보 보존과 LLM의 공간 추론 능력 모두의 한계에 기인한다.

**(4) 실시간 멀티모달 상호작용:**
음성 대화 중 시각 정보를 동시에 처리하는 실시간 상호작용은 연산 비용과 지연 시간의 엄격한 제약 하에서 해결해야 하는 공학적 과제이다.

### 4.7 벤치마크 종합 비교

주요 MLLM의 벤치마크 성능을 종합 비교하면 다음과 같다:

| 모델 | 규모 | MMBench | MMMU | MathVista | TextVQA | POPE | 비고 |
|------|------|---------|------|-----------|---------|------|------|
| LLaVA-1.5 | 13B | 67.7 | 36.4 | - | 61.3 | 85.9 | 오픈소스 기준선 |
| LLaVA-NeXT | 13B | 70.0 | - | - | 67.1 | - | 동적 해상도 |
| InternVL2 | 76B | 86.5 | 55.2 | 65.6 | 82.3 | - | 오픈소스 SOTA |
| Qwen2-VL | 72B | 86.9 | 64.5 | 70.5 | 85.5 | - | 다국어 강점 |
| MoAI | 7B | 77.2 | - | - | - | 87.1 | CV 모델 앙상블 |
| GPT-4o | - | 83.4 | 69.1 | 63.8 | - | - | 상용 SOTA |
| Gemini Ultra | - | - | 59.4 | 53.0 | 82.3 | - | 네이티브 멀티모달 |

이 표에서 주목할 점은 두 가지이다. 첫째, 오픈소스 모델(InternVL2, Qwen2-VL)이 상용 모델(GPT-4o)과 다수 벤치마크에서 동등하거나 우월한 성능을 보이고 있다는 것이다. 둘째, MMMU(대학 수준 멀티모달 추론)에서는 여전히 GPT-4o가 우위를 점하고 있어, 심층적 추론 능력에서의 격차가 존재한다는 것이다.

---

## 5. 참고 논문

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 1 | **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** | Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever | **ICML 2021** | https://arxiv.org/abs/2103.00020 |
| 2 | **Flamingo: a Visual Language Model for Few-Shot Learning** | Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan | **NeurIPS 2022** | https://arxiv.org/abs/2204.14198 |
| 3 | **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** | Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi | **ICML 2023** | https://arxiv.org/abs/2301.12597 |
| 4 | **Visual Instruction Tuning (LLaVA)** | Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee | **NeurIPS 2023** | https://arxiv.org/abs/2304.08485 |
| 5 | **Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** | Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee | arXiv 2023 | https://arxiv.org/abs/2310.03744 |
| 6 | **LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge** | Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee | arXiv 2024 | https://llava-vl.github.io/blog/2024-01-30-llava-next/ |
| 7 | **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks** | Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, Jifeng Dai | **CVPR 2024** | https://arxiv.org/abs/2312.14238 |
| 8 | **InternVL2: Better than the Best — Expanding Performance Boundaries of Open-Source Multimodal Models** | Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Gui-Song Xia, Wei Li, Junyi Chen, Qi Liu, Yifei Li, Xingcheng Zhang, Jianwei Yang, Hao Li, Chunyuan Li, Jifeng Dai, Yu Qiao, Dahua Lin, Xizhou Zhu | arXiv 2024 | https://arxiv.org/abs/2404.16821 |
| 9 | **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond** | Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou | arXiv 2023 | https://arxiv.org/abs/2308.12966 |
| 10 | **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** | Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, Junyang Lin | arXiv 2024 | https://arxiv.org/abs/2409.12191 |
| 11 | **MoAI: Mixture of All Intelligence for Large Language and Vision Models** | Byung-Kwan Lee, Beomchan Park, Chae Won Kim, Yong Man Ro | **ECCV 2024** | https://arxiv.org/abs/2403.07508 |
| 12 | **CoLLaVO: Crayon Large Language and Vision mOdel** | Byung-Kwan Lee, Beomchan Park, Chae Won Kim, Yong Man Ro | **NeurIPS 2024** | https://arxiv.org/abs/2402.11248 |
| 13 | **GPT-4V(ision) System Card** | OpenAI | OpenAI Technical Report 2023 | https://cdn.openai.com/papers/GPTV_System_Card.pdf |
| 14 | **GPT-4 Technical Report** | OpenAI | arXiv 2023 | https://arxiv.org/abs/2303.08774 |
| 15 | **Gemini: A Family of Highly Capable Multimodal Models** | Gemini Team, Google DeepMind | arXiv 2023 | https://arxiv.org/abs/2312.11805 |
| 16 | **Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context** | Gemini Team, Google DeepMind | arXiv 2024 | https://arxiv.org/abs/2403.05530 |
| 17 | **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection** | Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, Li Yuan | **EMNLP 2024** | https://arxiv.org/abs/2311.10122 |
| 18 | **LLaVA-Video: Video Instruction Tuning With Synthetic Data** | Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, Chunyuan Li | arXiv 2024 | https://arxiv.org/abs/2410.02713 |
| 19 | **SALMONN: Towards Generic Hearing Abilities for Large Language Models** | Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang | **ICLR 2024** | https://arxiv.org/abs/2310.13289 |
| 20 | **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** | Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, Jingren Zhou | arXiv 2023 | https://arxiv.org/abs/2311.07919 |
| 21 | **Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs** | Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, Saining Xie | **NeurIPS 2024** | https://arxiv.org/abs/2406.16860 |
| 22 | **Sigmoid Loss for Language Image Pre-Training (SigLIP)** | Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer | **ICCV 2023** | https://arxiv.org/abs/2303.15343 |
| 23 | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** | Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby | **ICLR 2021** | https://arxiv.org/abs/2010.11929 |
| 24 | **EVA-CLIP: Improved Training Techniques for CLIP at Scale** | Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, Yue Cao | arXiv 2023 | https://arxiv.org/abs/2303.15389 |
| 25 | **ShareGPT4V: Improving Large Multi-Modal Models with Better Captions** | Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, Dahua Lin | **ECCV 2024** | https://arxiv.org/abs/2311.12793 |
| 26 | **POPE: Polling-based Object Probing Evaluation for Object Hallucination** | Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen | **EMNLP 2023** | https://arxiv.org/abs/2305.10355 |
| 27 | **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark** | Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen | **CVPR 2024** | https://arxiv.org/abs/2311.16502 |
| 28 | **MMBench: Is Your Multi-modal Model an All-around Player?** | Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, Dahua Lin | **ECCV 2024** | https://arxiv.org/abs/2307.06281 |
| 29 | **Perceiver: General Perception with Iterative Attention** | Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira | **ICML 2021** | https://arxiv.org/abs/2103.03206 |
| 30 | **LLaMA: Open and Efficient Foundation Language Models** | Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample | arXiv 2023 | https://arxiv.org/abs/2302.13971 |
| 31 | **An Introduction to Vision-Language Modeling** | Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Diana Hazan, Yann LeCun, Nicolas Ballas | arXiv 2024 | https://arxiv.org/abs/2405.17247 |
| 32 | **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** | Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, Jianfeng Gao | **ICLR 2024** | https://arxiv.org/abs/2310.02255 |
| 33 | **DINOv2: Learning Robust Visual Features without Supervision** | Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski | **TMLR 2024** | https://arxiv.org/abs/2304.07193 |
| 34 | **PaLI-X: On Scaling up a Multilingual Vision and Language Model** | Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, AJ Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Peter Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut | **EMNLP 2024** | https://arxiv.org/abs/2305.18565 |
| 35 | **VideoChat2: Chat-Centric Video Understanding** | Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, Yu Qiao | **CVPR 2024** | https://arxiv.org/abs/2311.17005 |
| 36 | **The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)** | Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, Lijuan Wang | arXiv 2023 | https://arxiv.org/abs/2309.17421 |
| 37 | **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** | Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi | **ICML 2022** | https://arxiv.org/abs/2201.12086 |
