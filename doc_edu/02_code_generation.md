# 02. 코드 생성하기

## Python 코드 생성

```bash
# 간단한 함수
ollama run qwen2.5-coder:32b "Python으로 버블소트를 구현해줘"

# 클래스 설계
ollama run qwen2.5-coder:32b "Python으로 은행 계좌 클래스를 만들어줘. 입금, 출금, 잔액조회 메서드 포함"

# 디자인 패턴
ollama run qwen2.5-coder:32b "Python Observer 패턴 예제를 작성해줘"
```

## Go 코드 생성

```bash
# REST API
ollama run qwen2.5-coder:32b "Go언어로 간단한 REST API 서버를 만들어줘. /users GET, POST 엔드포인트"

# 동시성
ollama run qwen2.5-coder:32b "Go언어로 goroutine과 channel을 사용한 워커 풀을 구현해줘"

# 인터페이스
ollama run qwen2.5-coder:32b "Go언어로 Shape 인터페이스와 Circle, Rectangle 구현체를 만들어줘"
```

## 프롬프트 팁

### 1. 구체적으로 요구하기

```bash
# 나쁜 예
ollama run qwen2.5-coder:32b "서버 만들어줘"

# 좋은 예
ollama run qwen2.5-coder:32b "Go언어 net/http로 REST API를 만들어줘.
GET /users는 유저 목록 JSON 반환,
POST /users는 새 유저 생성,
인메모리 저장소 사용, 에러 처리 포함"
```

### 2. 코드만 요청하기

설명 없이 코드만 받고 싶을 때:

```bash
ollama run qwen2.5-coder:32b "Python으로 퀵소트를 구현해줘. 코드만 출력해."
```

### 3. 언어 지정하기

```bash
ollama run qwen2.5-coder:32b "한국어로 설명해줘. Go의 goroutine과 channel이 뭐야?"
```

## 생성된 코드를 파일로 저장

```bash
# 응답을 파일로 리다이렉트
ollama run qwen2.5-coder:32b "Python FastAPI CRUD 예제. 코드만 출력해." > app.py

# 확인
cat app.py
```
