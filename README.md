# myllm - 로컬 LLM 코딩 어시스턴트

로컬 환경에서 동작하는 LLM 기반 코딩 어시스턴트.
프로젝트 동작 원리 설명, Python/Go 등 코드 생성을 지원한다.

## 환경

| 항목 | 내용 |
|------|------|
| HW | MacBook Pro M4 Max, **128GB** 통합 메모리, 1.8TB SSD |
| 도구 | [Ollama](https://ollama.com/) v0.17.7 |
| 최적화 | Flash Attention, KV Cache q8_0 |

## 설치

```bash
# 1. Ollama 설치
brew install ollama

# 2. 서비스 시작 (로그인 시 자동 시작)
brew services start ollama

# 3. 모델 다운로드 (원하는 것 선택)
ollama pull qwen2.5-coder:32b                # 코딩 특화 32B (19GB)
ollama pull qwen2.5-coder:32b-instruct-q8_0  # 코딩 특화 32B 고품질 양자화 (34GB)
ollama pull qwen2.5:72b                      # 범용 72B (47GB)
ollama pull deepseek-coder-v2                # 코딩 경량 16B (8.9GB)
```

## 사용 가능한 모델

128GB 통합 메모리 기준으로 아래 모델을 모두 구동할 수 있다.

| 모델 | 파라미터 | 디스크 | 용도 | 비고 |
|------|---------|--------|------|------|
| `qwen2.5-coder:32b` | 32B | 19GB | 코딩 특화 | 기본 Q4 양자화, 빠름 |
| `qwen2.5-coder:32b-instruct-q8_0` | 32B | 34GB | 코딩 특화 | Q8 양자화, 더 정확 |
| `qwen2.5:72b` | 72B | 47GB | 범용 + 코딩 | 가장 큰 모델, 설명 능력 최상 |
| `deepseek-coder-v2` | 16B (MoE 236B) | 8.9GB | 코딩 특화 | 가볍고 빠름 |
| `llama3.1:70b` | 70B | 40GB | 범용 | Meta 오픈소스 |
| `codestral` | 22B | 12GB | 코딩 특화 | Mistral AI |

> 128GB에서는 72B 모델도 전체 GPU 오프로드가 가능하므로 속도 저하 없이 구동된다.
> 더 큰 모델(예: `qwen2.5:72b-instruct-q8_0` ~75GB)도 설치 가능하나 동시에 다른 작업 시 메모리 부족이 올 수 있다.

## 사용법

### 1. 터미널 대화형

```bash
# 한 줄 질문
ollama run qwen2.5-coder:32b "Python으로 퀵소트를 구현해줘"

# 대화 모드 진입 (종료: /bye)
ollama run qwen2.5-coder:32b

# 다른 모델 사용
ollama run qwen2.5:72b "이 알고리즘의 시간복잡도를 분석해줘"
```

### 2. 프로젝트 코드 분석

```bash
# 파일 하나를 전달하여 동작 원리 질문
cat main.py | ollama run qwen2.5-coder:32b "이 코드의 동작 원리를 설명해줘:"

# 여러 파일을 한꺼번에 전달
cat src/*.go | ollama run qwen2.5-coder:32b "이 프로젝트의 구조를 설명해줘:"

# 특정 함수에 대해 질문
grep -A 30 "func HandleRequest" server.go | ollama run qwen2.5-coder:32b "이 함수에 버그가 있어?"
```

### 3. REST API

Ollama는 `http://localhost:11434`에서 API를 제공한다.

```bash
# 단일 응답 (stream: false로 전체 응답을 한번에 받음)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "Go로 HTTP 서버를 만들어줘",
  "stream": false
}'

# 채팅 (멀티턴 대화)
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-coder:32b",
  "messages": [
    {"role": "system", "content": "너는 시니어 개발자야. 코드를 작성할 때 항상 에러 처리를 포함해."},
    {"role": "user", "content": "Python FastAPI REST API 예제를 만들어줘"}
  ],
  "stream": false
}'

# 대용량 컨텍스트로 긴 코드 분석 (128GB RAM 활용)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "이 코드를 분석해줘: ...(긴 코드)...",
  "stream": false,
  "options": {
    "num_ctx": 32768
  }
}'
```

### 4. OpenAI 호환 API

기존 OpenAI SDK/도구와 호환되는 엔드포인트를 제공한다.
`base_url`만 바꾸면 기존 코드를 그대로 사용할 수 있다.

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder:32b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Python에서 OpenAI SDK 사용:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen2.5-coder:32b",
    messages=[
        {"role": "system", "content": "너는 코딩 어시스턴트야."},
        {"role": "user", "content": "피보나치 함수를 작성해줘"},
    ],
)
print(response.choices[0].message.content)
```

### 5. 스트리밍 응답

실시간으로 토큰 단위 출력을 받을 수 있다.

```python
import json
import urllib.request

data = json.dumps({
    "model": "qwen2.5-coder:32b",
    "prompt": "Python으로 웹 크롤러를 만들어줘",
    "stream": True,
}).encode()

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=data,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    for line in resp:
        chunk = json.loads(line)
        print(chunk["response"], end="", flush=True)
        if chunk.get("done"):
            break
```

## 모델 관리

```bash
# 설치된 모델 목록 확인
ollama list

# 모델 상세 정보 확인
ollama show qwen2.5-coder:32b

# 새 모델 다운로드
ollama pull <모델명>

# 모델 삭제 (디스크 공간 회수)
ollama rm qwen2.5-coder:32b
ollama rm deepseek-coder-v2
ollama rm qwen2.5:72b

# 현재 메모리에 로드된 모델 확인
curl -s http://localhost:11434/api/ps | python3 -m json.tool

# 메모리에서 모델 언로드 (삭제는 아님)
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5-coder:32b","keep_alive":0}'
```

> 나중에 벤치마크 결과를 비교하여 최적 모델 1개만 남기고 나머지를 삭제할 수 있다.
> 삭제해도 `ollama pull`로 언제든 다시 받을 수 있다.

## 메모리 최적화 설정

### 동작 원리

M4 Max의 **통합 메모리 아키텍처** 덕분에 CPU와 GPU가 같은 메모리를 공유한다.
LLM 추론은 **GPU가 담당**하므로:

- Activity Monitor에서 CPU 사용량은 낮게 보임 (정상)
- GPU 사용량이 90%+ (정상, 이것이 쿨러가 도는 이유)
- 프로세스 메모리(RSS)가 작아 보이지만, 모델 가중치는 GPU 메모리 영역에 로드됨

### Ollama 환경변수

서비스 설정 파일: `~/Library/LaunchAgents/homebrew.mxcl.ollama.plist`

| 환경변수 | 값 | 설명 |
|----------|-----|------|
| `OLLAMA_FLASH_ATTENTION` | `1` | Flash Attention 활성화, 추론 속도 향상 |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | KV 캐시 양자화, 메모리 사용량 절감 |

### 런타임 옵션 (API 호출 시)

API의 `options` 필드로 전달:

| 옵션 | 기본값 | 128GB 권장 | 설명 |
|------|--------|-----------|------|
| `num_ctx` | 4096 | 16384~32768 | 컨텍스트 윈도우 크기. 길수록 긴 코드 분석 가능 |
| `temperature` | 0.8 | 0.1~0.3 | 코드 생성 시 낮을수록 정확 |
| `num_predict` | 128 | -1 | 최대 생성 토큰 수. -1은 무제한 |

```bash
# 예: 32K 컨텍스트 + 낮은 temperature로 정확한 코드 생성
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "...",
  "stream": false,
  "options": {
    "num_ctx": 32768,
    "temperature": 0.2,
    "num_predict": -1
  }
}'
```

## 서비스 관리

```bash
# 서비스 시작
brew services start ollama

# 서비스 중지
brew services stop ollama

# 서비스 재시작
brew services restart ollama

# 서비스 상태 확인
brew services info ollama

# 수동 실행 (디버깅 시)
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

## 테스트

```bash
# 기본 테스트 (서비스, 코드 생성, API 7개 항목)
python3 test_ollama.py

# 심화 테스트 (대용량 컨텍스트, 리팩토링, 버그 탐지, 성능 6개 항목)
python3 test_advanced.py

# 모델 비교 벤치마크 (7개 카테고리, 15개 시나리오)
# 설치된 모든 모델을 자동으로 비교 평가
# 결과는 experiments/results/ 에 JSON + Markdown으로 저장
python3 experiments/benchmark.py
```

## 벤치마크 결과

### qwen2.5-coder:32b (2026-03-13)

- 평균 점수: **92.8%**
- 평균 속도: **16.4 tokens/s**
- 평균 응답시간: **32.5s**

| 카테고리 | 시나리오 | 점수 |
|---------|---------|------|
| 알고리즘 | BST 구현 | 100% |
| 알고리즘 | 다익스트라 | 100% |
| 알고리즘 | LRU 캐시 | 100% |
| 웹/API | Go REST API | 100% |
| 웹/API | FastAPI CRUD | 100% |
| 웹/API | Go 미들웨어 | 100% |
| 디자인패턴 | Strategy | 100% |
| 디자인패턴 | Go Factory | 100% |
| 동시성 | Go 워커 풀 | 75% |
| 동시성 | Python 비동기 | 100% |
| 데이터처리 | CSV 파서 | 67% |
| 데이터처리 | Go JSON 스트림 | 100% |
| 디버깅 | 메모리 누수 분석 | 50% |
| 디버깅 | 레이스 컨디션 | 100% |
| 코드설명 | 데코레이터 설명 | 100% |

> 상세 결과: [experiments/results/](experiments/results/)

## 프로젝트 구조

```
myllm/
├── README.md                  # 이 문서
├── test_ollama.py             # 기본 테스트 (7개)
├── test_advanced.py           # 심화 테스트 (6개)
└── experiments/
    ├── benchmark.py           # 모델 비교 벤치마크 스크립트
    └── results/               # 벤치마크 결과 저장
        ├── benchmark_*.json   # 원본 데이터 (JSON)
        └── report_*.md        # 리포트 (Markdown)
```
