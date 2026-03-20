# RAG 구현 상세 — 구현 방법, 재연, 성능

## 1. 무엇을 구현했는가

대화가 256K 토큰을 초과해도 관련 과거 대화를 검색하여 답변할 수 있는 RAG(Retrieval Augmented Generation) 시스템.

```
기존: [전체 대화 기록] → Ollama → 256K 초과하면 실패
RAG:  [검색된 관련 대화 10개 + 최근 6턴] → Ollama → 항상 동작
```

### 구성 요소

```
rag_store.py          임베딩 생성/저장/검색 (sqlite3 + Ollama embed API)
migrate_history.py    기존 대화 기록 → 벡터 DB 인덱싱 (일회성)
web_chat.py           RAG 통합 웹 챗봇 서버
tmp/rag/vectors.db    벡터 DB (sqlite3)
```

### 기술 선택

| 항목 | 선택 | 이유 |
|------|------|------|
| 벡터 DB | sqlite3 | Python 내장, 설치 불필요, 수만 건 이하에서 충분 |
| 임베딩 모델 | nomic-embed-text (137MB) | Ollama에서 바로 사용, CPU에서도 빠름 |
| 유사도 | 코사인 유사도 | math 모듈만으로 구현 가능 |
| 외부 의존성 | 없음 | pip install 불필요 |

ChromaDB를 쓰지 않은 이유: numpy, onnxruntime 등 무거운 의존성 체인. 대화 기록 규모에서는 brute-force 검색으로 충분.

---

## 2. 어떻게 구현했는가

### 2.1 임베딩 생성 (`rag_store.py`)

Ollama의 `/api/embed` 엔드포인트로 텍스트를 768차원 벡터로 변환한다.

```python
def embed(self, text):
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({
            "model": "nomic-embed-text",
            "input": text
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    return result["embeddings"][0]  # 768차원 float 배열
```

### 2.2 청킹 전략

대화를 적절한 크기로 쪼개서 저장한다.

```
저장 단위: user+assistant 쌍을 하나로 묶음
  "질문: {user 메시지}\n답변: {assistant 메시지}"

청킹 기준:
  2000자 이하 → 그대로 1청크
  2000자 초과 → 단락(\n\n) 기준으로 분할, 200자 오버랩
```

오버랩이 필요한 이유: 청크 경계에서 맥락이 잘리는 것을 방지.

### 2.3 벡터 저장 (sqlite3)

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,       -- 세션명 (Tech, Mental, ...)
    role TEXT NOT NULL,          -- 'pair' (user+assistant 묶음)
    content TEXT NOT NULL,       -- 원문 텍스트
    embedding TEXT NOT NULL,     -- JSON으로 직렬화된 768차원 벡터
    turn_index INTEGER NOT NULL, -- 대화 순서 (몇 번째 턴인지)
    created_at REAL NOT NULL     -- 저장 시각
);

CREATE TABLE meta (
    session TEXT PRIMARY KEY,
    indexed_turns INTEGER NOT NULL DEFAULT 0  -- 인덱싱 완료된 턴 수
);
```

### 2.4 검색 (코사인 유사도)

```python
def search(self, query, session, top_k=10):
    query_vec = self.embed(query)  # 쿼리를 벡터로 변환

    rows = self.conn.execute(
        "SELECT content, embedding, turn_index FROM chunks WHERE session = ?",
        (session,)
    ).fetchall()

    # 모든 청크와 코사인 유사도 계산 (brute-force)
    scored = []
    for row in rows:
        vec = json.loads(row["embedding"])
        sim = cosine_similarity(query_vec, vec)
        scored.append({"content": row["content"], "score": sim, ...})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]  # 상위 10개 반환
```

코사인 유사도 공식:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)

A · B     = Σ(ai × bi)      두 벡터의 내적
||A||     = √Σ(ai²)         벡터의 크기
결과 범위  = -1.0 ~ 1.0      1에 가까울수록 유사
```

### 2.5 프롬프트 조립

RAG 모드일 때 Ollama에 보내는 메시지 구성:

```
┌─────────────────────────────────────────────────┐
│ 1. 시스템 프롬프트 (세션별 역할 설정)                │
├─────────────────────────────────────────────────┤
│ 2. RAG 검색 결과 (관련 과거 대화 top-10)            │
│    "다음은 이전 대화에서 현재 질문과 관련된 부분이다:     │
│     [대화 #3] 질문: ... 답변: ...                  │
│     [대화 #15] 질문: ... 답변: ...                 │
│     ..."                                         │
├─────────────────────────────────────────────────┤
│ 3. 최근 6턴 (직전 대화 맥락 유지)                    │
├─────────────────────────────────────────────────┤
│ 4. 현재 user 메시지                                │
└─────────────────────────────────────────────────┘
```

토큰 예산:

```
시스템 프롬프트:          ~200 토큰
RAG 검색 결과 (top-10):  ~8K 토큰
최근 6턴:                ~4K 토큰
응답 여유:               ~32K 토큰
───────────────────────────────────
총 사용량:               ~44K 토큰 (256K의 17%)
```

### 2.6 세션별 전략

```python
SESSION_PRESETS = {
    "Tech":   { "use_rag": True },    # 항상 RAG
    "Mental": { "use_rag": False },   # 기본은 전체 전송
}

# Mental도 200K 토큰 초과 시 자동으로 RAG 전환
if not use_rag and estimate_tokens(messages) >= 200000:
    use_rag = True
```

### 2.7 점진적 인덱싱

새 대화가 추가될 때마다 실시간으로 벡터 DB에 인덱싱한다.

```
사용자 질문 → Ollama 응답 → JSON 저장 → 벡터 DB 인덱싱
                                         ↑
                                    add_pair(session, user_msg, reply, turn_index)
```

서버 시작 시에도 JSON과 벡터 DB를 비교하여 누락분을 자동 인덱싱.

---

## 3. 재연 방법

### 3.1 환경 준비

```bash
# Ollama 시작
brew services start ollama

# 임베딩 모델 설치 (최초 1회, 137MB)
ollama pull nomic-embed-text

# 설치 확인
ollama list
# nomic-embed-text:latest    274 MB
# devstral-2:123b            75 GB
```

### 3.2 기존 대화 기록 인덱싱

```bash
# 수동 마이그레이션 (기존 tmp/chat/*.json → tmp/rag/vectors.db)
python3 migrate_history.py
```

출력 예시:

```
대화 기록 마이그레이션 시작...
  [상담] 전체 메시지: 2개, 인덱싱 완료: 0턴 → 1턴 새로 인덱싱

완료: 총 1턴 인덱싱됨
```

### 3.3 웹 챗봇 실행

```bash
python3 web_chat.py
```

출력 예시:

```
추론 속도 측정 중...
  GPU 비율: 81%
  프롬프트 처리: X tokens/sec
  생성 속도: 3.3 tokens/sec
  타임아웃: ...초
RAG 저장소 초기화 중...
  [상담] 1턴 인덱싱 완료
RAG 준비 완료.
MyLLM 웹 챗봇 시작!
  브라우저에서 열기: http://localhost:60000
```

### 3.4 RAG 동작 확인

```bash
# 벡터 DB 상태 확인
python3 -c "
from rag_store import RAGStore
store = RAGStore()
rows = store.conn.execute('SELECT session, COUNT(*) as cnt FROM chunks GROUP BY session').fetchall()
for r in rows:
    print(f'  {r[\"session\"]}: {r[\"cnt\"]}청크')
"
```

### 3.5 전체 초기화 (처음부터 다시)

```bash
# 벡터 DB만 삭제 (대화 기록은 유지)
rm -f tmp/rag/vectors.db

# 재인덱싱
python3 migrate_history.py
```

---

## 4. 성능 벤치마크 (M4 Max 실측)

### 4.1 임베딩 속도

nomic-embed-text 모델, CPU 처리.

| 입력 길이 | 소요 시간 | 비고 |
|-----------|----------|------|
| 짧은 문장 (17자) | 173ms | 첫 호출은 모델 로딩 포함 (웜업) |
| 보통 문장 (60자) | 18ms | 웜업 후 안정적 속도 |
| 긴 문장 (~3000자) | 47ms | 길어져도 크게 느려지지 않음 |

### 4.2 인덱싱 속도

| 규모 | 소요 시간 | 턴당 시간 |
|------|----------|----------|
| 100턴 | 1.7초 | 17ms/턴 |
| 1,000턴 추정 | ~17초 | 17ms/턴 |
| 10,000턴 추정 | ~170초 (~3분) | 17ms/턴 |

### 4.3 검색 속도

쿼리 1건의 검색 시간 (임베딩 생성 + 유사도 계산).

| 청크 수 | 소요 시간 | 비고 |
|---------|----------|------|
| 100개 | 27ms | 즉시 |
| 500개 | 68ms | 즉시 |
| 1,000개 추정 | ~120ms | 체감 없음 |
| 10,000개 추정 | ~1초 | 여전히 빠름 |
| 100,000개 추정 | ~10초 | 이 시점에서 최적화 필요 |

brute-force 검색의 한계: 10만 청크 이상이면 FAISS 등 ANN(Approximate Nearest Neighbor) 라이브러리가 필요하다. 일반적 대화 기록에서는 도달하기 어려운 규모.

### 4.4 저장 용량

| 규모 | 벡터 DB 크기 | 비고 |
|------|-------------|------|
| 500턴 | 6.0MB | |
| 1,000턴 추정 | ~12MB | |
| 10,000턴 추정 | ~120MB | |
| 100,000턴 추정 | ~1.2GB | SSD에서 문제없음 |

벡터 1개 = 768 × 8바이트(float64 JSON) ≈ 6KB. 청크 텍스트 포함하면 ~12KB/턴.

### 4.5 전체 응답 파이프라인

"Tech 세션에서 질문 1건"의 소요 시간 분해:

```
단계                          시간         비율
────────────────────────────────────────────────
① 쿼리 임베딩                ~20ms        0.1%
② 벡터 검색 (500청크 기준)    ~50ms        0.3%
③ 프롬프트 조립               ~1ms         0%
④ Ollama 프롬프트 처리        ~30초        15%
⑤ Ollama 토큰 생성           ~3분         84%
⑥ JSON/DB 저장               ~20ms        0%
────────────────────────────────────────────────
RAG 오버헤드 (①+②+③+⑥)     ~91ms        0.4%
LLM 추론 (④+⑤)              ~3.5분       99.6%
```

RAG 때문에 추가되는 시간은 전체의 0.4% 미만. 사실상 체감 차이 없음.

### 4.6 처리 가능한 규모

| 항목 | 값 | 제한 요소 |
|------|-----|----------|
| 최대 대화 턴 수 | 사실상 무제한 | 디스크 용량 |
| 검색 가능 청크 수 | ~10만 (1초 이내) | CPU 연산 |
| 한 번에 참조 가능한 과거 대화 | top-10 (~8K 토큰) | 설정값 조절 가능 |
| 최대 프롬프트 크기 | ~44K 토큰 (256K의 17%) | 모델 컨텍스트 |
| 한 번 응답의 최대 길이 | ~32K 토큰 | 모델 제한 |

---

## 5. 한계와 개선 가능성

### 현재 한계

| 한계 | 설명 |
|------|------|
| brute-force 검색 | 10만 청크 이상에서 느려짐 |
| 단일 임베딩 모델 | 한글 특화 임베딩 모델이 아님 |
| 청크 경계 | 오버랩으로 완화하지만 완벽하지 않음 |
| 검색 품질 | 의미적 유사도만 사용, 키워드 매칭 없음 |

### 개선 가능 방향

| 개선 | 방법 |
|------|------|
| 검색 속도 | FAISS 또는 hnswlib 도입 (10만+ 청크) |
| 검색 품질 | 하이브리드 검색 (벡터 + BM25 키워드) |
| 한글 성능 | 한글 특화 임베딩 모델 (multilingual-e5 등) |
| 요약 기억 | 긴 대화를 주기적으로 요약하여 별도 저장 |
