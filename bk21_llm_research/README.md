# LLM 기법 조사 보고서 — 해외 탑 컨퍼런스 중심

> BK21 학회 랭킹 기준 해외 탑 컨퍼런스(NeurIPS, ICML, ICLR, ACL, CVPR 등) 발표 논문 중심으로 LLM 기법을 조사한 문서 모음

## 디렉토리 구조

```
bk21_llm_research/
├── README.md                           # 본 파일
├── techniques/                         # LLM 핵심 기법 14개 (각 기법 상세 + 논문 참고문헌)
│   ├── 01_fine_tuning.md              # Fine-tuning (LoRA, QLoRA, PEFT, Instruction Tuning)
│   ├── 02_rag.md                      # RAG (검색 증강 생성)
│   ├── 03_prompt_engineering.md       # Prompt Engineering (CoT, ToT, ReAct, DSPy)
│   ├── 04_rlhf.md                    # RLHF / DPO / GRPO (정렬 기법)
│   ├── 05_knowledge_distillation.md   # Knowledge Distillation (지식 증류)
│   ├── 06_quantization.md            # Quantization (GPTQ, AWQ, GGUF, BitNet)
│   ├── 07_multimodal_llm.md          # Multi-modal LLM (VLM)
│   ├── 08_pretraining.md             # Pre-training (Scaling Laws, 분산학습)
│   ├── 09_attention_mechanism.md     # Attention & Transformer (GQA, RoPE, Flash Attention)
│   ├── 10_in_context_learning.md     # In-Context Learning
│   ├── 11_chain_of_thought.md        # Chain-of-Thought & Reasoning Models (o1, R1)
│   ├── 12_mixture_of_experts.md      # Mixture of Experts (MoE)
│   ├── 13_alignment.md              # AI Alignment & Safety
│   └── 14_efficient_inference.md     # Efficient Inference (vLLM, Speculative Decoding)
├── conferences/                       # 학회 정보
│   ├── conference_rankings.md        # BK21 기준 학회 랭킹 (KAIST/SNU/POSTECH 등)
│   └── korean_conferences.md         # 주요 학회 상세 정보 및 LLM 논문 목록
└── korean_llm/                        # 한국어 LLM 모델 정보
    └── korean_llm_models.md          # 한국어 LLM 모델 총정리
```

## 문서 구성

각 기법 문서는 다음 구조를 따릅니다:

1. **역사적 배경과 문제 인식** — 왜 필요한가? 이전에 어떤 문제가 있었나?
2. **진화 과정** — 기법의 발전 타임라인 (어떤 문제 → 어떤 해결 → 어떤 새 문제)
3. **기법 상세** — 핵심 원리, 수식, 장단점 비교
4. **장단점 종합** — 기법별 장점/단점 정리
5. **참고 논문** — 탑 컨퍼런스 논문 테이블 (저자, 학회, arXiv 링크)

## 참조 학회 등급 (BK21 기준)

| 등급 | 학회 예시 |
|------|-----------|
| **최우수** | NeurIPS, ICML, ICLR, ACL, CVPR, ICCV, AAAI, SOSP, OSDI, ISCA |
| **우수** | EMNLP, NAACL, COLING, ECCV, MLSys, AISTATS, IJCAI |

## 총 수록 논문 수

약 **150+편**의 탑 컨퍼런스/arXiv 논문을 참고문헌으로 수록

## 작성일

2026-03-29
