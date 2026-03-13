# 03. 코드 분석 및 디버깅

## 파일 분석하기

내 프로젝트의 코드를 LLM에게 전달하여 분석을 요청할 수 있다.

### 단일 파일 분석

```bash
# 파이프로 파일 내용을 전달
cat main.py | ollama run qwen2.5-coder:32b "이 코드의 동작 원리를 설명해줘:"

# 또는 특정 함수만 추출하여 전달
grep -A 50 "def process_data" app.py | ollama run qwen2.5-coder:32b "이 함수를 분석해줘:"
```

### 여러 파일 분석

```bash
# 프로젝트 전체 구조 설명 요청
cat src/*.py | ollama run qwen2.5-coder:32b "이 프로젝트의 구조와 각 모듈의 역할을 설명해줘:"

# Go 프로젝트 분석
cat *.go | ollama run qwen2.5-coder:32b "이 Go 프로젝트의 아키텍처를 설명해줘:"
```

## 버그 찾기

```bash
cat buggy_code.py | ollama run qwen2.5-coder:32b "이 코드에서 버그를 찾아서 수정해줘. 이유도 설명해:"
```

## 코드 리팩토링

```bash
cat messy_code.py | ollama run qwen2.5-coder:32b "이 코드를 리팩토링해줘.
- 가독성 향상
- 중복 제거
- 함수 분리"
```

## 코드 리뷰

```bash
# git diff를 전달하여 코드 리뷰 받기
git diff | ollama run qwen2.5-coder:32b "이 변경사항을 코드 리뷰해줘. 문제점이나 개선할 점이 있으면 알려줘:"

# 특정 커밋의 변경사항 리뷰
git show HEAD | ollama run qwen2.5-coder:32b "이 커밋을 리뷰해줘:"
```

## 보안 분석

```bash
cat server.py | ollama run qwen2.5-coder:32b "이 코드에서 보안 취약점을 찾아줘.
SQL 인젝션, XSS, 인증 문제 등을 확인해:"
```

## 테스트 코드 생성

```bash
cat user_service.py | ollama run qwen2.5-coder:32b "이 코드에 대한 pytest 단위 테스트를 작성해줘:"
```
