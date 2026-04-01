# 한국어 LLM 모델 총정리

## 1. 개요

한국어 특화 LLM은 한국어 코퍼스로 사전 학습되었거나, 한국어에 대한 추가 학습(Continual Pre-training, Fine-tuning)을 수행한 대규모 언어 모델이다. 2023년 이후 오픈소스 한국어 LLM이 급증하고 있다.

## 2. 주요 한국어 LLM 모델

### 2.1 기업 개발 모델 (상용/준오픈소스)

| 모델 | 개발사 | 크기 | 라이선스 | 특징 |
|------|--------|------|----------|------|
| **HyperCLOVA X** | NAVER | 비공개 | 비공개 | 한국어 최적화, CLOVA 서비스 탑재 |
| **EXAONE 3.0/3.5** | LG AI Research | 7.8B | Apache 2.0 | 한국어-영어 bilingual |
| **Solar** | Upstage | 10.7B | Apache 2.0 | DUS(Depth Up-Scaling) 기법 |
| **VARCO** | NCSOFT | 다양 | 제한적 | 게임/대화 특화 |
| **AYA** | Cohere for AI | 다양 | 오픈 | 다국어(한국어 포함) |

### 2.2 오픈소스 한국어 LLM

| 모델 | 기반 모델 | 크기 | 링크 | 특징 |
|------|-----------|------|------|------|
| **Polyglot-Ko** | GPT-NeoX | 1.3B~12.8B | https://huggingface.co/EleutherAI/polyglot-ko-12.8b | 한국어 처음부터 학습 |
| **KULLM (구름)** | LLaMA | 5.8B~12.8B | https://github.com/nlpai-lab/KULLM | 고려대 NLP랩 |
| **KoAlpaca** | LLaMA | 7B~13B | https://github.com/Beomi/KoAlpaca | 한국어 Alpaca 데이터 |
| **KORani** | LLaMA-2 | 13B | - | 한국어 지시 학습 |
| **RedWhale** | - | 다양 | https://arxiv.org/abs/2408.11294 | 한국어 처음부터 학습 |
| **OPEN-SOLAR-KO** | Solar | 10.7B | - | Solar 기반 한국어 강화 |
| **komt** | LLaMA-2 | 7B~13B | - | 한국어 지시 학습 |
| **Synatra** | Mistral | 7B | - | Mistral 기반 한국어 |
| **EEVE-Korean** | Mistral | 10.8B | - | 한국어 어휘 확장 |
| **Bllossom** | LLaMA-3 | 8B | - | 한국어 Continual PT |

### 2.3 모델 상세

#### HyperCLOVA X (NAVER)
- **논문**: https://clova.ai/en/tech-blog/introducing-hyperclova-x-our-state-of-the-art-ai-models-optimized-for-the-korean-language
- **학회**: ACL 2024 (Technical Report)
- **특징**:
  - 한국어-영어-코드 데이터로 사전 학습
  - 한국 문화/법률/상식에 특화
  - CLOVA X, CLOVA Note 등 서비스에 적용

#### EXAONE (LG AI Research)
- **논문**: https://arxiv.org/abs/2408.03541
- **특징**:
  - Expert AI for Everyone
  - 한국어-영어 bilingual 사전 학습
  - Apache 2.0 라이선스로 오픈소스 공개
  - MoE 아키텍처 (EXAONE 3.5)

#### Solar (Upstage)
- **특징**:
  - DUS (Depth Up-Scaling): 기존 모델을 깊이 방향으로 확장
  - Mistral 7B 기반 10.7B로 확장
  - 한국어 벤치마크 상위권

#### Polyglot-Ko (EleutherAI Korea)
- **논문**: https://arxiv.org/abs/2306.02254
- **특징**:
  - 한국어 863GB 데이터로 처음부터 학습
  - GPT-NeoX 아키텍처 기반
  - 1.3B, 3.8B, 5.8B, 12.8B 크기 제공
  - 한국어 오픈소스 LLM의 시초

#### KULLM 구름 (고려대학교)
- **GitHub**: https://github.com/nlpai-lab/KULLM
- **개발**: 고려대학교 NLP&AI Lab
- **특징**:
  - Korea University Large Language Model
  - GPT-4를 활용한 한국어 지시 데이터 생성
  - BK21 FOUR 고려대학교 교육연구단 관련

#### RedWhale
- **논문**: https://arxiv.org/abs/2408.11294
- **특징**:
  - 한국어에 최적화된 토크나이저 설계
  - 한국어 데이터 비율을 높인 사전 학습
  - 효율적인 한국어 처리

## 3. 한국어 LLM 벤치마크

### 3.1 주요 벤치마크

| 벤치마크 | 설명 | 링크 |
|----------|------|------|
| **KMMLU** | 한국어 Massive Multitask Language Understanding | https://arxiv.org/abs/2402.11548 |
| **KoBEST** | 한국어 NLU 벤치마크 (BoolQ, COPA, WiC, HellaSwag, SentiNeg) | https://huggingface.co/datasets/skt/kobest_v1 |
| **KorNAT** | 한국어 자연어 추론 | - |
| **CLIcK** | 한국 문화 및 언어 이해 벤치마크 | - |
| **HAE-RAE Bench** | 한국어 종합 평가 | - |
| **Ko-Chatbot Arena** | 한국어 챗봇 인간 평가 | - |

### 3.2 KMMLU (Korean MMLU)
- **논문**: https://arxiv.org/abs/2402.11548
- **구성**: 45개 한국어 과목 (인문, 사회, 과학, 공학, 의학 등)
- **의의**: 한국어 LLM의 표준 평가 도구

## 4. 한국어 데이터셋

| 데이터셋 | 설명 | 용도 |
|----------|------|------|
| **KIT-19** | 19개 태스크 한국어 지시 데이터 | Instruction Tuning |
| **Korean ShareGPT** | 한국어 대화 데이터 | SFT |
| **KoAlpaca 데이터** | GPT-3.5 기반 한국어 지시 데이터 | Instruction Tuning |
| **AIHub 한국어 데이터** | 정부 지원 한국어 NLP 데이터 | 다양한 NLP 태스크 |
| **Kor-NLI/STS** | 한국어 자연어 추론/의미 유사도 | 평가 |

## 5. 참고 자료

### GitHub 리소스
- **Awesome Korean LLM**: https://github.com/NomaDamas/awesome-korean-llm
  - 한국어 LLM 모델, 데이터셋, 벤치마크 종합 목록

### 관련 논문
1. Polyglot-Ko: https://arxiv.org/abs/2306.02254
2. KMMLU: https://arxiv.org/abs/2402.11548
3. RedWhale: https://arxiv.org/abs/2408.11294
4. EXAONE 3.0: https://arxiv.org/abs/2408.03541
5. KIT-19: https://aclanthology.org/2024.lrec-main.296/
6. HyperCLOVA X: ACL 2024 Technical Report
