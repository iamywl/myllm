# myllm - 로컬 LLM 코딩 어시스턴트

로컬 환경에서 동작하는 LLM 기반 코딩 어시스턴트.

## 환경

| 항목 | 내용 |
|------|------|
| HW | MacBook Pro M4 Max, **128GB** 통합 메모리, 1.8TB SSD |
| 도구 | [Ollama](https://ollama.com/) |
| 모델 | **Devstral-2 123B** (Q4_K_M, 75GB) |
| 성능 | SWE-bench Verified 72.2% (오픈소스 코딩 모델 1위) |
| 컨텍스트 | 256K 토큰 |

## 설치

```bash
# 1. Ollama 설치
brew install ollama

# 2. 서비스 시작
brew services start ollama

# 3. Devstral-2 설치 (기존 모델 삭제 + 75GB 다운로드)
./install_devstral2.sh
```

---

## 빠른 시작

### 전체 흐름: 시작 → 사용 → 확인 → 종료

```bash
# 1. Ollama 시작
brew services start ollama

# 2. 사용 (아래 방법 중 택1)
ollama run devstral-2:123b                  # 터미널 대화
./chat.sh Tech                              # 터미널 채팅 (맥락 기억)
python3 web_chat.py                         # 웹 챗봇

# 3. 모델이 메모리에 올라가 있는지 확인
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# 4. 작업 끝나면 모델 내리기 (126GB 메모리 해제)
curl http://localhost:11434/api/generate -d '{"model":"devstral-2:123b","keep_alive":0}'

# 5. Ollama까지 완전히 끄기 (선택)
brew services stop ollama
```

---

## 사용법

### 터미널

```bash
# 한 줄 질문
ollama run devstral-2:123b "Python으로 퀵소트를 구현해줘"

# 대화 모드 (종료: /bye)
ollama run devstral-2:123b

# 파일 전달해서 코드 분석
cat main.py | ollama run devstral-2:123b "이 코드의 버그를 찾아줘"

# 여러 파일 한번에
cat src/*.go | ollama run devstral-2:123b "이 프로젝트 구조를 설명해줘"

# 임시 파일에 분석할 내용을 작성한 뒤 LLM에 전달
cat <<'EOF' > tmp/analyze.txt
여기에 분석할 코드나 텍스트를 붙여넣기
EOF
cat tmp/analyze.txt | ollama run devstral-2:123b "이 내용을 분석해줘"

# 또는 한 줄로
echo "분석할 내용" > tmp/q.txt && cat tmp/q.txt | ollama run devstral-2:123b "분석해줘"

# 분석 결과를 파일로 저장
cat tmp/analyze.txt | ollama run devstral-2:123b "이 코드를 리뷰해줘" > tmp/result.txt
```

### 채팅 모드 (대화 맥락 기억)

```bash
# 새 세션 시작
./chat.sh 상담

# 입력 후 빈 줄(Enter)로 전송, quit로 종료
# 파일 내용을 보내려면 @경로 입력
나> @tmp/q.txt
나> 위 분석에서 2번 항목을 더 자세히 설명해줘
나> quit
```

- 대화 기록이 `tmp/chat/세션명.json`에 자동 저장됨
- 같은 세션명으로 다시 실행하면 이전 대화를 이어감
- 세션명 없이 `./chat.sh`만 실행하면 `default` 세션 사용

### 웹 챗봇

```bash
# 1. 서버 시작 (시작 시 GPU 기반 타임아웃 자동 측정)
python3 web_chat.py

# 2. 브라우저에서 접속
open http://localhost:60000
```

- **Tech 세션**: 기술 상담. RAG로 모든 대화 기록을 검색하여 관련 맥락을 자동 참조
- **Mental 세션**: 심리 상담. 공감 중심 대화 (전체 맥락 전송, 256K 초과 시 RAG 자동 전환)
- 세션 전환은 상단 드롭다운에서 선택
- 새 세션을 만들려면 세션명 입력 후 "새 세션" 클릭
- 대화 기록은 `tmp/chat/`에 저장 (터미널 채팅과 공유, git 추적 안 함)
- RAG 벡터 DB는 `tmp/rag/`에 저장 (git 추적 안 함)

#### 기존 대화 기록 인덱싱

서버 시작 시 자동으로 인덱싱하지만, 수동 실행도 가능하다.

```bash
python3 migrate_history.py
```

### REST API

```bash
# 단일 응답
curl http://localhost:11434/api/generate -d '{
  "model": "devstral-2:123b",
  "prompt": "Go로 HTTP 서버를 만들어줘",
  "stream": false
}'

# 채팅 (멀티턴)
curl http://localhost:11434/api/chat -d '{
  "model": "devstral-2:123b",
  "messages": [
    {"role": "system", "content": "너는 시니어 개발자야."},
    {"role": "user", "content": "Python FastAPI REST API 예제를 만들어줘"}
  ],
  "stream": false
}'

# 대용량 컨텍스트 (256K까지 가능)
curl http://localhost:11434/api/generate -d '{
  "model": "devstral-2:123b",
  "prompt": "이 코드를 분석해줘: ...",
  "stream": false,
  "options": {"num_ctx": 65536}
}'
```

### OpenAI 호환 API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="devstral-2:123b",
    messages=[
        {"role": "system", "content": "너는 코딩 어시스턴트야."},
        {"role": "user", "content": "피보나치 함수를 작성해줘"},
    ],
)
print(response.choices[0].message.content)
```

---

## 상태 확인

```bash
# 모델이 메모리에 올라가 있는지 확인
curl -s http://localhost:11434/api/ps | python3 -m json.tool
```

- `"models": []` → 모델이 메모리에 **없음** (해제된 상태)
- `"models": [...]` → 모델이 메모리에 **올라가 있음**
  - `size`: 모델 전체 크기 (~126GB)
  - `size_vram`: GPU에 올라간 양 (~102GB, 81%)
  - 나머지 ~24GB는 CPU/RAM에서 처리 (19%)

```bash
# Ollama 서비스 상태 확인
brew services info ollama

# 설치된 모델 목록
ollama list

# 모델 상세 정보
ollama show devstral-2:123b
```

---

## 종료 (작업 후 반드시 확인)

123B 모델은 **~126GB 메모리**를 점유한다. 작업이 끝나면 반드시 해제해야 한다.

### 단계별 종료

```bash
# 1단계: 웹 챗봇 종료 (사용한 경우)
#   - 웹 UI에서 "서버 종료" 버튼 클릭
#   - 또는 터미널에서 Ctrl+C

# 2단계: 모델을 메모리에서 내리기
curl http://localhost:11434/api/generate -d '{"model":"devstral-2:123b","keep_alive":0}'

# 3단계: 해제 확인 (models가 빈 배열이면 완료)
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# 4단계 (선택): Ollama 서비스 자체를 종료
brew services stop ollama
```

### 종료 방법 비교

| 방법 | 명령어 | 모델 메모리 | Ollama 서비스 | 다시 시작하려면 |
|------|--------|------------|--------------|----------------|
| 모델만 내리기 | `curl ...keep_alive:0` | 해제 | 유지 | 바로 사용 가능 (자동 재로드) |
| Ollama 종료 | `brew services stop ollama` | 해제 | 종료 | `brew services start ollama` |
| 웹 챗봇만 종료 | Ctrl+C 또는 UI 버튼 | **유지됨** | 유지 | `python3 web_chat.py` |

> **주의**: 웹 챗봇을 종료해도 모델은 메모리에 남아있다. 반드시 2단계(모델 내리기)를 해야 126GB가 해제된다.

---

## 벤치마크

```bash
python3 benchmark.py
```

## 모델 관리

```bash
# 설치된 모델 확인
ollama list

# 모델 상세 정보
ollama show devstral-2:123b

# 모델 삭제 (디스크에서 완전 제거, 75GB 확보)
ollama rm devstral-2:123b
```

## 메모리 최적화

### Ollama 환경변수

| 환경변수 | 값 | 설명 |
|----------|-----|------|
| `OLLAMA_FLASH_ATTENTION` | `1` | Flash Attention 활성화 |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV 캐시 양자화, 메모리 절감 |

### API 옵션

| 옵션 | 기본값 | 권장값 | 설명 |
|------|--------|--------|------|
| `num_ctx` | 4096 | 32768~65536 | 컨텍스트 윈도우. 256K까지 가능 |
| `temperature` | 0.8 | 0.1~0.3 | 코드 생성 시 낮을수록 정확 |
| `num_predict` | 128 | -1 | 최대 생성 토큰. -1은 무제한 |

## Ollama 서비스 관리

```bash
brew services start ollama    # 시작
brew services stop ollama     # 종료
brew services restart ollama  # 재시작
brew services info ollama     # 상태 확인
```

## 프로젝트 구조

```
myllm/
├── README.md              # 이 문서
├── install_devstral2.sh   # 설치 스크립트
├── chat.sh                # 채팅 스크립트 (대화 맥락 기억)
├── web_chat.py            # 웹 챗봇 서버 (RAG 통합)
├── rag_store.py           # RAG 임베딩 저장소 모듈
├── migrate_history.py     # 대화 기록 → 벡터 DB 마이그레이션
├── benchmark.py           # 성능 벤치마크
├── doc_edu/               # 교육 문서
├── experiments/           # 실험
└── tmp/                   # 임시 파일, 대화 기록, 벡터 DB (git 추적 안 함)
```
