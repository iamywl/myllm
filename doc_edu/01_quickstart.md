# 01. 빠르게 시작하기

## Ollama란?

Ollama는 로컬 환경에서 LLM(대규모 언어 모델)을 쉽게 실행할 수 있게 해주는 도구다.
클라우드 API 없이 내 맥북에서 바로 AI 모델을 돌릴 수 있다.

## 서비스 확인

```bash
# Ollama가 실행 중인지 확인
brew services info ollama

# 실행 중이 아니면 시작
brew services start ollama
```

## 첫 번째 질문 해보기

```bash
# 터미널에서 바로 질문
ollama run qwen2.5-coder:32b "Python으로 Hello World를 출력해줘"
```

출력 예시:
```
print("Hello World")
```

## 대화 모드

```bash
# 대화 모드 진입
ollama run qwen2.5-coder:32b
```

진입하면 프롬프트가 나타난다:
```
>>> 피보나치 함수를 만들어줘
(답변 출력)

>>> 그거를 재귀 말고 반복문으로 바꿔줘
(답변 출력)

>>> /bye
```

`/bye`를 입력하면 대화 모드에서 나온다.

## 설치된 모델 확인

```bash
ollama list
```

출력 예시:
```
NAME                        SIZE      MODIFIED
deepseek-coder-v2:latest    8.9 GB    1 hour ago
qwen2.5-coder:32b           19 GB     1 hour ago
```
