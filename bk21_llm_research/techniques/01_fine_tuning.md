# Fine-tuning (미세 조정)

---

## 1. 기법의 정의

Fine-tuning은 대규모 텍스트 코퍼스에서 사전 학습(Pre-training)을 완료한 언어 모델의 파라미터를 특정 도메인 또는 하위 태스크(downstream task)에 맞게 추가 학습하는 기법이다. 사전 학습 단계에서 획득한 범용 언어 표현을 초기값으로 활용함으로써, 소규모 태스크 데이터만으로도 높은 성능을 달성할 수 있다.

Fine-tuning은 크게 두 범주로 구분된다:
- **Full Fine-tuning**: 모델의 전체 파라미터를 업데이트하는 방식이다.
- **Parameter-Efficient Fine-Tuning (PEFT)**: 모델 파라미터의 극소 일부만 학습하여 연산 및 메모리 비용을 절감하는 방식이다.

---

## 2. 기존 기법의 한계와 Fine-tuning 등장 배경

### 2.1 사전 학습 모델의 한계

사전 학습(Pre-training) 단계에서 모델은 대규모 코퍼스에 대한 다음 토큰 예측(Causal Language Modeling) 또는 마스킹 토큰 복원(Masked Language Modeling)을 수행한다. 이 과정에서 모델은 문법, 상식, 세계 지식 등 범용적 언어 표현을 획득하지만, 다음과 같은 한계가 존재한다:

1. **태스크 특이성 부재**: 사전 학습의 목적 함수는 범용적 언어 모델링이므로, 감성 분석, 질의응답, 코드 생성 등 특정 태스크에 대한 최적화가 이루어지지 않는다.
2. **도메인 지식 부족**: 의료, 법률, 금융 등 전문 도메인의 용어와 추론 패턴은 일반 웹 텍스트에 충분히 반영되어 있지 않다.
3. **지시 수행 능력 부재**: 사전 학습만 수행한 모델은 "다음 단어 예측"만 학습하였으므로, 사용자의 자연어 지시(instruction)를 해석하고 이에 따라 응답을 생성하는 능력이 없다.

### 2.2 처음부터 학습하는 방식의 비용 문제

특정 태스크를 위해 모델을 처음부터(scratch) 학습하는 것은 비실용적이다. GPT-3 (175B) 학습에 약 $4.6M, LLaMA-3 (405B) 학습에 약 30.84M GPU-hours가 소요되었다. 따라서 사전 학습된 가중치를 초기값으로 재활용하여 소규모 데이터로 추가 학습하는 Fine-tuning이 필수적인 기법으로 자리잡았다.

### 2.3 Full Fine-tuning의 한계와 PEFT의 등장

Full Fine-tuning은 BERT (110M~340M) 규모에서는 실용적이었으나, LLM 규모가 수십~수백 B 파라미터로 증가하면서 다음과 같은 문제가 발생하였다:

1. **메모리 요구량**: 7B 모델의 Full Fine-tuning에는 약 60GB 이상의 GPU VRAM이 필요하다. 이는 모델 파라미터(FP16: ~14GB), 옵티마이저 상태(AdamW: ~28GB), 그래디언트(~14GB), 활성값의 합산이다.
2. **태스크별 모델 사본**: N개의 태스크에 대해 Full Fine-tuning을 수행하면, N개의 전체 모델 사본을 저장해야 한다. 70B 모델 기준 사본 1개당 ~140GB이다.
3. **Catastrophic Forgetting**: 소규모 데이터로 전체 파라미터를 업데이트하면, 사전 학습 단계에서 획득한 범용 지식이 손실될 수 있다.

이러한 한계를 해결하기 위해 **Parameter-Efficient Fine-Tuning (PEFT)** 계열의 기법이 제안되었다.

---

## 3. 주요 기법 상세

### 3.1 Full Fine-tuning

모델의 전체 파라미터 θ를 태스크 데이터셋 D = {(x_i, y_i)}에 대해 업데이트하는 방식이다.

**목적 함수:**
```
θ* = argmin_θ Σ L(f_θ(x_i), y_i)
```

여기서 L은 태스크에 따른 손실 함수(Cross-Entropy, MSE 등)이다.

**장점:**
- 태스크 특화 성능의 상한이 가장 높다.
- 모델의 모든 레이어가 태스크에 적응한다.

**단점:**
- GPU 메모리 요구량이 모델 크기의 약 4배(FP16 기준 AdamW 사용 시)이다.
- 태스크마다 전체 모델 사본이 필요하다.
- 소규모 데이터에서 과적합 및 catastrophic forgetting 위험이 존재한다.

**실험 결과 참고:**
Hu et al. (2021)의 LoRA 논문에서 보고된 바에 따르면, GPT-3 175B에 대한 Full Fine-tuning은 1.2TB 이상의 체크포인트 저장 공간이 필요하며, 학습 중 약 1.2TB의 GPU 메모리가 요구된다.

### 3.2 Adapter Tuning

**등장 배경:**
Full Fine-tuning의 메모리 문제를 해결하기 위해 Houlsby et al. (2019)이 ICML 2019에서 제안한 기법이다. 핵심 아이디어는 Transformer 블록 내부에 소규모 "Adapter" 모듈을 삽입하고, 원본 파라미터는 고정(freeze)한 채 Adapter 파라미터만 학습하는 것이다.

**구조:**
```
Adapter(x) = x + f(x · W_down) · W_up
- W_down ∈ R^{d×r}: 다운 프로젝션 (d → r, r << d)
- W_up ∈ R^{r×d}: 업 프로젝션 (r → d)
- f: 비선형 활성 함수 (ReLU)
- 잔차 연결(residual connection)로 안정성 확보
```

**학습 파라미터 비율:** 전체의 약 1~3%

**한계:**
- Adapter 모듈이 Transformer 블록 내에 순차적으로 삽입되므로, **추론 시 추가적인 연산 지연(latency)**이 발생한다. Hu et al. (2021)의 측정에 따르면, 단일 GPU에서 배치 크기 1 추론 시 Adapter는 Full model 대비 약 20~30%의 지연이 추가된다.
- 이 추론 지연 문제가 이후 LoRA 개발의 직접적 동기가 되었다.

### 3.3 Prefix Tuning

**등장 배경:**
Li & Liang (2021)은 ACL 2021에서 Prefix Tuning을 제안하였다. Adapter의 추론 지연 문제를 회피하면서도, 모델 파라미터를 고정하고 소규모 파라미터만 학습하는 것이 목표였다.

**핵심 아이디어:**
입력 시퀀스 앞에 학습 가능한 "가상 토큰(virtual tokens)"의 연속적 벡터를 접두사(prefix)로 추가한다. 이 prefix는 어휘(vocabulary)에 대응하는 실제 토큰이 아니라, 연속적인 벡터 공간에서 자유롭게 최적화되는 파라미터이다.

```
입력: [prefix_1, ..., prefix_m, x_1, ..., x_n]
- prefix_i ∈ R^d: 학습 가능한 연속 벡터
- x_j: 실제 입력 토큰의 임베딩
```

**학습 파라미터 비율:** 전체의 약 0.1%

**한계:**
- prefix 길이가 길어지면 모델의 유효 컨텍스트 길이가 감소한다.
- 복잡한 태스크에서 LoRA 대비 성능이 낮은 경향이 있다(Hu et al., 2021).
- prefix 벡터가 해석 불가능하여 디버깅이 어렵다.

### 3.4 LoRA (Low-Rank Adaptation)

**등장 배경:**
Hu et al. (2021)은 기존 PEFT 기법들의 다음 한계를 지적하였다:
- Adapter: 추론 시 추가 지연이 발생한다.
- Prefix Tuning: 유효 시퀀스 길이가 감소하며, 최적화가 어렵다.
- 두 기법 모두 Full Fine-tuning 대비 성능 격차가 존재한다.

또한 Aghajanyan et al. (2020)의 연구에서 사전 학습된 모델의 파라미터가 태스크 적응 시 **낮은 내재 차원(low intrinsic dimensionality)**을 가진다는 실험적 증거가 제시되었다. 이는 파라미터 업데이트량 ΔW가 저랭크(low-rank) 행렬로 근사 가능하다는 것을 의미한다.

**핵심 수식:**
```
h = W₀x + ΔWx = W₀x + BAx

여기서:
- W₀ ∈ R^{d×k}: 원본 사전 학습 가중치 (고정, freeze)
- B ∈ R^{d×r}: 저랭크 행렬 (학습 대상)
- A ∈ R^{r×k}: 저랭크 행렬 (학습 대상)
- r << min(d, k): 랭크, 보통 r ∈ {4, 8, 16, 32, 64}
- 학습 파라미터 수: r(d + k), 원본 대비 약 0.01~0.5%
```

**초기화 전략:**
- A는 정규분포 N(0, σ²)로 초기화한다.
- B는 영행렬(zero matrix)로 초기화한다.
- 따라서 학습 초기에 ΔW = BA = 0이므로, 원본 모델과 동일한 상태에서 출발한다.
- 스케일링 팩터 α/r을 곱하여 학습률을 조절한다.

**적용 위치:**
Hu et al. (2021)의 실험에서 Transformer의 어텐션 가중치(W_q, W_v)에 LoRA를 적용하는 것이 가장 효과적이었다. 이후 연구에서는 W_k, W_o, FFN 가중치에도 적용하여 성능을 추가 향상시키는 것이 보고되었다.

**추론 시 병합:**
```
W' = W₀ + BA
```
학습 완료 후 BA를 원본 가중치에 병합하면, 추론 시 추가적인 연산이나 메모리 오버헤드가 전혀 없다. 이것이 Adapter 대비 LoRA의 핵심 장점이다.

**실험 결과 (Hu et al., 2021):**

| 모델 | 방법 | 학습 파라미터 | MNLI (acc) | SST-2 (acc) | MRPC (acc) |
|------|------|-------------|-----------|------------|-----------|
| RoBERTa-base | Full FT | 125M (100%) | 87.6 | 94.8 | 90.2 |
| RoBERTa-base | LoRA (r=8) | 0.3M (0.24%) | 87.5 | 95.1 | 89.7 |
| GPT-3 175B | Full FT | 175B (100%) | - | - | - |
| GPT-3 175B | LoRA (r=4) | 4.7M (0.003%) | - | - | - |

LoRA는 0.003%의 파라미터만 학습하면서도 Full Fine-tuning과 동등하거나 유사한 성능을 달성하였다.

**장점:**
- 추론 시 추가 지연이 없다 (병합 가능).
- 여러 태스크용 LoRA 어댑터를 독립적으로 저장하고 교체 사용할 수 있다 (LoRA adapter = 수 MB~수십 MB).
- 학습 메모리가 Full Fine-tuning 대비 약 3배 감소한다.
- 원본 모델 가중치가 보존되므로 catastrophic forgetting 위험이 감소한다.

**단점:**
- 랭크 r의 선택이 휴리스틱에 의존한다. 최적 r은 태스크와 모델에 따라 다르다.
- Full Fine-tuning 대비 표현력의 이론적 상한이 존재한다. 특히 사전 학습 분포와 크게 다른 태스크에서 성능 격차가 벌어질 수 있다.
- 모든 레이어에 동일한 r을 적용하는 것은 비최적적이다 (이후 AdaLoRA에서 개선).

### 3.5 QLoRA (Quantized LoRA)

**등장 배경:**
LoRA가 학습 파라미터 수를 대폭 줄였으나, 원본 모델은 여전히 FP16 (16-bit)으로 GPU 메모리에 로딩되어야 하였다. 65B 모델의 경우 FP16 로딩에만 약 130GB가 필요하므로, 단일 GPU에서의 Fine-tuning이 불가능하였다.

Dettmers et al. (2023)은 NeurIPS 2023에서 QLoRA를 제안하여, **4-bit 양자화된 모델 위에서 LoRA 학습을 수행**함으로써 단일 48GB GPU에서 65B 모델의 Fine-tuning을 가능하게 하였다.

**3가지 핵심 기술 혁신:**

**(1) 4-bit NormalFloat (NF4):**
사전 학습된 모델 가중치의 분포가 정규분포 N(0, σ²)를 따른다는 경험적 관찰에 기반하여, 정보 이론적으로 최적인 4-bit 양자화 데이터 타입을 설계하였다. NF4는 가중치 분포의 각 양자화 빈(bin)에 동일한 확률 질량을 할당함으로써, 4-bit 양자화에서의 정보 손실을 최소화한다.

**(2) Double Quantization (이중 양자화):**
블록 단위 양자화에서 사용되는 양자화 상수(quantization constants)를 다시 양자화하여 메모리를 추가 절감한다. 64개 파라미터당 1개의 FP32 양자화 상수(32 bits)를 8-bit로 재양자화하면, 파라미터당 약 0.37 bits의 추가 절감 효과가 있다.

**(3) Paged Optimizers:**
NVIDIA의 Unified Memory를 활용하여, GPU 메모리가 부족할 때 옵티마이저 상태를 자동으로 CPU RAM으로 페이징한다. 이를 통해 GPU OOM(Out-Of-Memory) 오류 없이 학습을 지속할 수 있다.

**실험 결과 (Dettmers et al., 2023):**
- Guanaco-65B (QLoRA로 학습): 단일 48GB GPU에서 24시간 학습, Chatbot Arena에서 ChatGPT (GPT-3.5) 수준의 성능을 달성하였다.
- 4-bit QLoRA는 16-bit Full Fine-tuning과 통계적으로 유의미한 성능 차이가 없었다 (MMLU 기준).
- 메모리 사용량: 65B 모델 기준 Full FT ~780GB → QLoRA ~48GB (약 16배 감소).

**한계:**
- 4-bit 역양자화(dequantization) 연산으로 인해 학습 속도가 FP16 LoRA 대비 약 30~40% 느리다.
- NF4 양자화로 인한 미세한 정보 손실이 존재하며, 이는 일부 정밀한 태스크에서 성능 저하로 이어질 수 있다.

### 3.6 LoRA 후속 변형 기법

LoRA 이후 다수의 변형 기법이 제안되었다. 주요 기법을 시간순으로 정리한다.

**(1) AdaLoRA (Zhang et al., 2023, ICLR 2023):**
- **기존 문제:** LoRA는 모든 레이어에 동일한 랭크 r을 할당한다. 그러나 레이어별 중요도는 상이하므로, 이는 비최적적인 파라미터 분배이다.
- **해결:** SVD 기반의 파라미터 중요도 점수를 산출하여, 중요한 레이어에 높은 랭크를 동적으로 할당한다. 학습 과정에서 불필요한 특잇값(singular value)을 가지치기한다.
- **효과:** 동일 파라미터 예산에서 LoRA 대비 평균 1.2% 성능 향상이 보고되었다.

**(2) DoRA (Shih-Yang Liu et al., 2024, ICML 2024):**
- **기존 문제:** LoRA의 저랭크 업데이트 ΔW = BA는 가중치의 크기(magnitude)와 방향(direction)을 동시에 변경한다. 분석 결과, Full Fine-tuning은 주로 방향(direction)을 변경하는 반면, LoRA는 크기와 방향을 동시에 변경하여 학습 효율이 낮다.
- **해결:** Weight Decomposition을 도입하여 가중치를 크기(magnitude) m과 방향(direction) V로 분해한 후, LoRA를 방향 성분에만 적용한다.
```
W' = m · (V + BA/||V + BA||)
```
- **효과:** LoRA 대비 0.5~1.5% 성능 향상. 추가 파라미터 오버헤드는 무시할 수 있는 수준이다.

**(3) LoRA+ (Hayou et al., 2024, ICML 2024):**
- **기존 문제:** LoRA의 행렬 A와 B에 동일한 학습률을 적용하는 것은 비최적적이다. 이론적 분석에 의하면, A와 B의 최적 학습률 비율은 모델 너비(width)에 비례한다.
- **해결:** B의 학습률을 A의 학습률보다 크게 설정한다 (η_B = λ · η_A, λ ≈ 16).
- **효과:** LoRA 대비 최대 2% 성능 향상. 구현 변경이 학습률 설정 1줄에 불과하다.

**(4) rsLoRA (Kalajdzievski, 2023):**
- **기존 문제:** LoRA의 스케일링 팩터 α/r은 랭크 r이 증가하면 LoRA의 기여가 지나치게 커진다.
- **해결:** 스케일링 팩터를 α/√r로 변경하여 랭크에 대한 스케일 불변성을 확보한다.

**(5) GaLore (Zhao et al., 2024, ICML 2024):**
- **기존 문제:** LoRA는 저랭크 부분공간에서만 학습하므로, Full Fine-tuning 대비 표현력 제한이 존재한다.
- **해결:** Gradient Low-Rank Projection을 통해 Full Fine-tuning의 그래디언트를 저랭크 부분공간에 투영하여 메모리를 절감한다. LoRA와 달리 전체 파라미터 공간에서의 학습이 가능하다.
- **효과:** Full Fine-tuning 수준의 성능을 LoRA 수준의 메모리로 달성한다.

### 3.7 Instruction Tuning

**등장 배경:**
사전 학습 + LoRA/Full FT만으로는 모델이 사용자의 자연어 지시(instruction)를 이해하고 이에 따라 응답을 생성하는 능력이 부족하였다. 사전 학습 목적 함수는 "다음 토큰 예측"이므로, 모델은 텍스트 연속(continuation)만 수행할 뿐 지시 수행(instruction following)은 학습하지 않는다.

**핵심 구조:**
학습 데이터를 (instruction, input, output) 트리플렛으로 구성한다:
```
Instruction: "다음 텍스트를 요약하시오."
Input: "인공지능은 컴퓨터 과학의 한 분야로..."
Output: "AI는 인간 지능을 모사하는 컴퓨터 과학 분야이다."
```

**주요 연구 흐름:**

| 연구 | 학회 | 태스크 수 | 핵심 기여 |
|------|------|-----------|-----------|
| FLAN (Wei et al., 2021) | ICLR 2022 | 62 | Instruction Tuning 최초 대규모 실증 |
| T0 (Sanh et al., 2022) | ICLR 2022 | 36 | 프롬프트 기반 멀티태스크 학습 |
| Super-NaturalInstructions (Wang et al., 2022) | EMNLP 2022 | 1,616 | 대규모 태스크 확장 |
| Flan-T5/PaLM (Chung et al., 2022) | - | 1,836 | Scaling + CoT 데이터 결합 |
| InstructGPT (Ouyang et al., 2022) | NeurIPS 2022 | - | RLHF와 결합하여 ChatGPT의 기반 |
| Self-Instruct (Wang et al., 2023) | ACL 2023 | 자동 생성 | 모델이 스스로 학습 데이터 생성 |
| Alpaca (Taori et al., 2023) | - | 52K | GPT-3.5로 데이터 생성 → LLaMA 학습 |

**Self-Instruct (Wang et al., 2023, ACL 2023):**
- **기존 문제:** 고품질 Instruction 데이터의 수동 구축은 비용이 높다.
- **해결:** 모델 자체가 seed instructions로부터 새로운 instruction-output 쌍을 자동 생성한다.
- **절차:** (1) Seed 태스크 정의 → (2) LLM이 새 instruction 생성 → (3) LLM이 해당 instruction에 대한 output 생성 → (4) 품질 필터링 → (5) Student 모델 학습
- **의의:** 이후 Alpaca, Vicuna 등 오픈소스 Instruction Tuning 연구의 기반이 되었다.

### 3.8 PEFT 기법 종합 비교

| 기법 | 제안 학회 | 학습 파라미터 비율 | 추론 오버헤드 | Full FT 대비 성능 | 핵심 한계 |
|------|-----------|-------------------|-------------|------------------|-----------|
| Full Fine-tuning | - | 100% | 없음 | 기준 | 메모리, 저장 공간 |
| Adapter (Houlsby, 2019) | ICML 2019 | 1~3% | 있음 (지연) | -1~2% | 추론 지연 |
| Prefix Tuning (Li, 2021) | ACL 2021 | ~0.1% | 없음 | -2~3% | 유효 길이 감소 |
| Prompt Tuning (Lester, 2021) | EMNLP 2021 | ~0.01% | 없음 | -3~5% | 대형 모델에서만 효과 |
| LoRA (Hu, 2021) | ICLR 2022 | 0.01~0.5% | 없음 (병합) | -0~1% | 랭크 선택 휴리스틱 |
| IA³ (Liu, 2022) | NeurIPS 2022 | ~0.01% | 없음 | -1~2% | 표현력 제한 |
| QLoRA (Dettmers, 2023) | NeurIPS 2023 | 0.01~0.5% | 없음 (병합) | -0~1% | 학습 속도 저하 |
| AdaLoRA (Zhang, 2023) | ICLR 2023 | 0.01~0.5% (적응) | 없음 | +0.5~1.5% vs LoRA | 학습 복잡도 증가 |
| DoRA (Liu, 2024) | ICML 2024 | 0.01~0.5% | 없음 | +0.5~1.5% vs LoRA | 분해 연산 오버헤드 |
| GaLore (Zhao, 2024) | ICML 2024 | 전체 (투영) | 없음 | ≈ Full FT | 투영 재계산 비용 |

---

## 4. 기법 진화의 인과적 흐름

```
[문제] 사전학습 모델이 특정 태스크에 비최적
  ↓
[해결] Full Fine-tuning (BERT, 2018)
  ↓
[새 문제] LLM 규모 증가로 Full FT의 메모리/비용 비실용적
  ↓
[해결] Adapter (Houlsby, 2019, ICML)
  ↓
[새 문제] Adapter의 추론 시 추가 지연
  ↓
[해결] LoRA (Hu, 2021, ICLR 2022) — 추론 시 병합으로 지연 제거
  ↓
[새 문제] LoRA도 FP16 모델 로딩 필요 → 대형 모델 단일 GPU 불가
  ↓
[해결] QLoRA (Dettmers, 2023, NeurIPS) — 4-bit 양자화 + LoRA
  ↓
[새 문제] 동일 랭크 할당의 비최적성, LoRA의 표현력 제한
  ↓
[해결] AdaLoRA (동적 랭크), DoRA (방향 분해), GaLore (전체 공간 학습)
  ↓
[병행] Instruction Tuning으로 "지시 수행" 능력 부여
  ↓
[병행] RLHF/DPO로 인간 선호도에 정렬 (→ 04_rlhf.md 참조)
```

---

## 5. 참고 논문

### 5.1 핵심 논문 (Primary References)

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 1 | LoRA: Low-Rank Adaptation of Large Language Models | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen | **ICLR 2022** | https://arxiv.org/abs/2106.09685 |
| 2 | QLoRA: Efficient Finetuning of Quantized Language Models | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer | **NeurIPS 2023** | https://arxiv.org/abs/2305.14314 |
| 3 | Parameter-Efficient Transfer Learning for NLP (Adapter) | Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly | **ICML 2019** | https://arxiv.org/abs/1902.00751 |
| 4 | Prefix-Tuning: Optimizing Continuous Prompts for Generation | Xiang Lisa Li, Percy Liang | **ACL 2021** | https://arxiv.org/abs/2101.00190 |
| 5 | The Power of Scale for Parameter-Efficient Prompt Tuning | Brian Lester, Rami Al-Rfou, Noah Constant | **EMNLP 2021** | https://arxiv.org/abs/2104.08691 |
| 6 | Training language models to follow instructions with human feedback (InstructGPT) | Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. | **NeurIPS 2022** | https://arxiv.org/abs/2203.02155 |
| 7 | Scaling Instruction-Finetuned Language Models (Flan-T5/PaLM) | Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. | arXiv 2022 | https://arxiv.org/abs/2210.11416 |
| 8 | Self-Instruct: Aligning Language Models with Self-Generated Instructions | Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi | **ACL 2023** | https://arxiv.org/abs/2212.10560 |
| 9 | Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (IA³) | Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, Colin Raffel | **NeurIPS 2022** | https://arxiv.org/abs/2205.05638 |
| 10 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova | **NAACL 2019** | https://arxiv.org/abs/1810.04805 |

### 5.2 후속 및 변형 논문

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 11 | AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao | **ICLR 2023** | https://arxiv.org/abs/2303.10512 |
| 12 | DoRA: Weight-Decomposed Low-Rank Adaptation | Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen | **ICML 2024** | https://arxiv.org/abs/2402.09353 |
| 13 | LoRA+: Efficient Low Rank Adaptation of Large Models | Soufiane Hayou, Nikhil Ghosh, Bin Yu | **ICML 2024** | https://arxiv.org/abs/2402.12354 |
| 14 | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian | **ICML 2024** | https://arxiv.org/abs/2403.03507 |
| 15 | LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models | Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia | **ICLR 2024** | https://arxiv.org/abs/2309.12307 |
| 16 | Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning | Armen Aghajanyan, Sonal Gupta, Luke Zettlemoyer | **ACL 2021** | https://arxiv.org/abs/2012.13255 |
| 17 | Alpaca: A Strong, Replicable Instruction-Following Model | Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto | Stanford 2023 | https://crfm.stanford.edu/2023/03/13/alpaca.html |
| 18 | Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality | Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, Eric P. Xing | LMSYS 2023 | https://lmsys.org/blog/2023-03-30-vicuna/ |
| 19 | Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks | Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, et al. | **EMNLP 2022** | https://arxiv.org/abs/2204.07705 |
| 20 | Finetuned Language Models Are Zero-Shot Learners (FLAN) | Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le | **ICLR 2022** | https://arxiv.org/abs/2109.01652 |

### 5.3 서베이 및 벤치마크

| # | 논문 | 저자 | 학회 | 링크 |
|---|------|------|------|------|
| 21 | Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning | Vladislav Lialin, Vijeta Deshpande, Anna Rumshisky | arXiv 2023 | https://arxiv.org/abs/2303.15647 |
| 22 | A Survey on Efficient Fine-tuning Methods of Large Language Models | Xingchen Wan, Ruoxi Sun, Hanjun Dai, Sercan Ö. Arik, Tomas Pfister | arXiv 2024 | https://arxiv.org/abs/2401.04679 |
| 23 | LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of LLMs | Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, Roy Ka-Wei Lee | **EMNLP 2023** | https://arxiv.org/abs/2304.01933 |
| 24 | LoRA Learns Less and Forgets Less | Dan Biderman, Jose Gonzalez Ortiz, Jacob Portes, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham | arXiv 2024 | https://arxiv.org/abs/2405.09673 |
| 25 | rsLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | Damjan Kalajdzievski | arXiv 2023 | https://arxiv.org/abs/2312.03732 |
| 26 | MELoRA: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei | **ACL 2024** | https://arxiv.org/abs/2402.17263 |
| 27 | Towards a Unified View of Parameter-Efficient Transfer Learning | Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig | **ICLR 2022** | https://arxiv.org/abs/2110.04366 |
| 28 | Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models | Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. | arXiv 2022 | https://arxiv.org/abs/2203.06904 |

### 5.4 참고 라이브러리

| 이름 | 링크 | 설명 |
|------|------|------|
| HuggingFace PEFT | https://huggingface.co/docs/peft | LoRA, AdaLoRA, Prefix Tuning 등 PEFT 통합 라이브러리 |
| Unsloth | https://github.com/unslothai/unsloth | LoRA/QLoRA 학습 2배 속도 향상, 메모리 60% 절감 |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory | 다양한 Fine-tuning 기법 통합 프레임워크 |
| Axolotl | https://github.com/OpenAccess-AI-Collective/axolotl | 다양한 PEFT 기법 + 분산 학습 지원 |
| TRL | https://huggingface.co/docs/trl | SFT, RLHF, DPO 학습 라이브러리 |
