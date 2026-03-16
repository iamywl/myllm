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
cat <<'EOF' > /tmp/analyze.txt
여기에 분석할 코드나 텍스트를 붙여넣기
EOF
cat /tmp/analyze.txt | ollama run devstral-2:123b "이 내용을 분석해줘"

# 또는 한 줄로
echo "분석할 내용" > /tmp/q.txt && cat /tmp/q.txt | ollama run devstral-2:123b "분석해줘"

# 분석 결과를 파일로 저장
cat /tmp/analyze.txt | ollama run devstral-2:123b "이 코드를 리뷰해줘" > /tmp/result.txt
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

# 메모리에 로드된 모델 확인
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# 메모리에서 언로드 (삭제 아님)
curl http://localhost:11434/api/generate -d '{"model":"devstral-2:123b","keep_alive":0}'

# 모델 삭제
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

## 서비스 관리

```bash
brew services start ollama    # 시작
brew services stop ollama     # 중지
brew services restart ollama  # 재시작
brew services info ollama     # 상태 확인
```

## 프로젝트 구조

```
myllm/
├── README.md              # 이 문서
├── install_devstral2.sh   # 설치 스크립트
├── benchmark.py           # 성능 벤치마크
├── doc_edu/               # 문서
└── experiments/            # 실험
```
