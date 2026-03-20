# RAG 구현 계획

> 대화가 256K 토큰을 초과해도 관련 맥락만 검색해서 답변할 수 있게 한다.

## 왜 RAG가 필요한가?

```
현재 방식 (전체 전송):
  [대화 1] + [대화 2] + ... + [대화 500] + [새 질문]  →  Ollama
                                                        ↑
                                              256K 토큰 초과하면 실패!

RAG 방식 (검색 후 전송):
  [새 질문으로 검색] → 관련 대화 10개만 추출 → [관련 대화] + [최근 대화] + [새 질문]  →  Ollama
                                                                                     ↑
                                                                           항상 적은 토큰!
```

## 아키텍처

```
사용자 질문
    │
    ▼
① 임베딩 생성 (nomic-embed-text, ~137MB)
    │
    ▼
② 벡터 DB 검색 (sqlite3, 코사인 유사도)
    │  → 관련 과거 대화 top-10 추출
    │
    ▼
③ 프롬프트 조립
    │  시스템 프롬프트
    │  + "관련 과거 대화" (검색 결과)
    │  + 최근 6턴 (직전 맥락)
    │  + 현재 질문
    │
    ▼
④ Ollama에 전송 (Devstral-2 123B)
    │
    ▼
⑤ 응답 저장 + 새 턴을 벡터 DB에 인덱싱
```

### 왜 ChromaDB가 아닌 sqlite3인가?

- ChromaDB는 numpy, onnxruntime 등 무거운 의존성 체인
- 대화 기록 규모는 수만 건 이하 → brute-force 검색으로 충분
- Python 내장 sqlite3만으로 외부 설치 없이 동작

## 파일 구조

```
myllm/
├── web_chat.py          # 수정: RAG 통합
├── rag_store.py         # 신규: 임베딩 저장소 모듈
├── migrate_history.py   # 신규: 기존 대화 마이그레이션
└── tmp/
    ├── chat/            # 기존: JSON 대화 기록 (원본)
    └── rag/
        └── vectors.db   # 신규: sqlite 벡터 DB (검색 인덱스)
```

## 단계별 구현 계획

---

### 단계 1: 임베딩 인프라 구축 (`rag_store.py`)

**사전 작업**: 임베딩 모델 설치

```bash
ollama pull nomic-embed-text    # 137MB 다운로드
```

**구현할 것**: `rag_store.py` — 임베딩 생성/저장/검색 모듈

```python
# 핵심 구조
class RAGStore:
    def __init__(self):
        # sqlite3 DB 생성
        # 테이블: chunks(id, session, role, content, embedding_json, turn_index, timestamp)

    def embed(self, text) -> list[float]:
        # Ollama /api/embed 호출 → 벡터 반환

    def add_turn(self, session, role, content, turn_index):
        # content를 청크로 분할 (2000자 초과 시)
        # 각 청크를 임베딩하여 sqlite에 저장

    def search(self, query, session, top_k=10) -> list[dict]:
        # query를 임베딩
        # 해당 session의 모든 벡터와 코사인 유사도 계산
        # 상위 k개 반환
```

**청킹 전략**:
- 기본 단위: user+assistant 쌍 (질문+답변을 하나로)
- 긴 응답(2000자 초과): 단락(`\n\n`) 기준으로 분할
- 청크 간 200자 오버랩으로 맥락 유실 방지

**기술 결정**:
- 임베딩 모델: `nomic-embed-text` (CPU에서도 빠름, 수 밀리초/청크)
- 벡터 저장: sqlite3 (Python 내장, 설치 불필요)
- 유사도: 코사인 유사도 (math 모듈로 구현)
- 1만 개 청크 기준 검색 시간: 수십 밀리초 이내

**예상 작업량**: 약 150줄, 2~3시간

---

### 단계 2: 기존 대화 마이그레이션 (`migrate_history.py`)

기존 `tmp/chat/*.json`의 대화 기록을 벡터 DB로 인덱싱하는 일회성 스크립트.

```bash
python3 migrate_history.py
# tmp/chat/*.json → tmp/rag/vectors.db
```

서버 시작 시에도 자동 체크:
```python
# web_chat.py 시작 시
if not store.is_migrated(session):
    store.migrate_from_json(history_file)
```

**예상 작업량**: 약 40줄, 1시간

---

### 단계 3: `web_chat.py` RAG 통합

**현재 흐름**:
```
1. JSON에서 전체 messages 로드
2. user 메시지 추가
3. 전체 messages를 Ollama에 전송 ← 여기가 문제!
4. 응답 저장
```

**변경 후 흐름**:
```
1. JSON에서 전체 messages 로드
2. user 메시지 추가
3. [RAG] user 메시지로 벡터 검색 → 관련 과거 대화 top-10 추출
4. [RAG] 프롬프트 조립:
    - 시스템 프롬프트
    - 관련 과거 대화 (검색 결과)
    - 최근 6턴 (직전 맥락 유지)
    - 현재 user 메시지
5. 조립된 messages를 Ollama에 전송
6. 응답을 JSON에 저장
7. [RAG] 새 user+assistant 턴을 벡터 DB에 인덱싱
```

**토큰 예산**:
```
시스템 프롬프트:        ~200 토큰
RAG 검색 결과 (top-10): ~8K 토큰
최근 6턴:               ~4K 토큰
응답 여유:              ~32K 토큰
─────────────────────────────────
총 사용량:              ~44K 토큰 (256K의 17%, 충분한 여유)
```

**세션별 전략**:
```python
SESSION_PRESETS = {
    "Tech":   { ..., "use_rag": True },    # RAG 사용 (무한 기억)
    "Mental": { ..., "use_rag": False },   # 전체 전송 (짧은 대화)
}

# Mental도 토큰 한도 초과 시 자동으로 RAG 전환
if not use_rag and estimate_tokens(messages) < 200000:
    # 기존 방식
else:
    # RAG 방식
```

**예상 작업량**: 약 80줄 수정, 2~3시간

---

### 단계 4: 테스트 및 튜닝

- top_k 값 조정 (5 vs 10 vs 20)
- 청크 크기 조정 (1000자 vs 2000자)
- 검색 품질 확인: "저번에 말한 그 API"가 잘 검색되는지
- 응답 품질 비교: RAG vs 전체 전송

**예상 작업량**: 1~2시간

---

## 전체 일정

| 단계 | 내용 | 예상 시간 | 의존성 |
|------|------|-----------|--------|
| 0 | `ollama pull nomic-embed-text` | 5분 | - |
| 1 | `rag_store.py` 임베딩 인프라 + 청킹 | 2~3시간 | 단계 0 |
| 2 | `migrate_history.py` 마이그레이션 | 1시간 | 단계 1 |
| 3 | `web_chat.py` RAG 통합 | 2~3시간 | 단계 1, 2 |
| 4 | 테스트 및 튜닝 | 1~2시간 | 단계 3 |
| **총계** | | **6~9시간** | |

## 핵심 포인트

1. **외부 의존성 제로**: pip install 없이 Python 내장 + Ollama만 사용
2. **JSON은 그대로 유지**: 벡터 DB는 검색 인덱스일 뿐, JSON이 원본(source of truth)
3. **임베딩은 CPU 처리**: GPU가 Devstral-2로 점유되어 있어도 임베딩은 CPU로 충분
4. **점진적 인덱싱**: 새 대화가 추가될 때마다 실시간 인덱싱 (마이그레이션은 최초 1회만)
5. **Tech 세션만 RAG**: Mental 세션은 기존 방식 유지 (짧은 대화 특성)
