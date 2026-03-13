# 04. API로 사용하기

Ollama는 `http://localhost:11434`에서 REST API를 제공한다.
프로그램에서 LLM을 호출하고 싶을 때 사용한다.

## API 엔드포인트 정리

| 엔드포인트 | 용도 |
|-----------|------|
| `POST /api/generate` | 단일 프롬프트 → 응답 |
| `POST /api/chat` | 멀티턴 대화 |
| `GET /api/tags` | 설치된 모델 목록 |
| `GET /api/ps` | 현재 로드된 모델 |
| `POST /v1/chat/completions` | OpenAI 호환 API |

## 1. 단일 응답 받기 (curl)

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "Python으로 피보나치 함수를 작성해줘",
  "stream": false
}'
```

응답 JSON에서 `response` 필드가 LLM의 답변이다.

## 2. 멀티턴 대화 (curl)

이전 대화 내용을 `messages`에 포함하여 전달한다.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-coder:32b",
  "messages": [
    {"role": "system", "content": "너는 시니어 Go 개발자야."},
    {"role": "user", "content": "HTTP 서버를 만들어줘"},
    {"role": "assistant", "content": "아래는 간단한 HTTP 서버입니다..."},
    {"role": "user", "content": "여기에 미들웨어를 추가해줘"}
  ],
  "stream": false
}'
```

## 3. Python에서 사용하기 (urllib)

외부 라이브러리 설치 없이 사용할 수 있다.

```python
import json
import urllib.request

def ask_llm(prompt, model="qwen2.5-coder:32b"):
    """로컬 LLM에게 질문하기"""
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
        return result["response"]


# 사용 예시
answer = ask_llm("Python으로 이진 탐색을 구현해줘")
print(answer)
```

## 4. Python OpenAI SDK로 사용하기

기존 OpenAI SDK 코드를 거의 그대로 재활용할 수 있다.

```bash
pip install openai
```

```python
from openai import OpenAI

# base_url만 로컬로 변경
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="unused",  # 로컬이므로 아무 값이나 넣으면 됨
)

response = client.chat.completions.create(
    model="qwen2.5-coder:32b",
    messages=[
        {"role": "system", "content": "너는 코딩 어시스턴트야. 한국어로 답변해."},
        {"role": "user", "content": "Go로 gRPC 서버 예제를 만들어줘"},
    ],
)

print(response.choices[0].message.content)
```

## 5. 옵션 조정하기

API 호출 시 `options`로 모델 동작을 제어할 수 있다.

```python
data = {
    "model": "qwen2.5-coder:32b",
    "prompt": "...",
    "stream": False,
    "options": {
        "temperature": 0.2,   # 낮을수록 정확 (코드 생성 시 0.1~0.3 권장)
        "num_ctx": 32768,     # 컨텍스트 크기 (긴 코드 분석 시 확대)
        "num_predict": -1,    # 최대 토큰 수 (-1 = 무제한)
        "top_p": 0.9,         # 샘플링 범위
    },
}
```

| 옵션 | 코드 생성 권장값 | 설명 |
|------|----------------|------|
| `temperature` | 0.1 ~ 0.3 | 낮을수록 결정적, 높을수록 창의적 |
| `num_ctx` | 16384 ~ 32768 | 입력+출력 총 토큰 수. 긴 코드 분석 시 확대 |
| `num_predict` | -1 | 생성할 최대 토큰. -1은 제한 없음 |
| `top_p` | 0.9 | nucleus sampling. 보통 기본값 사용 |
