# 06. 실전 팁 & 자주 묻는 질문

## 실전 활용 예시

### 1. 새 프로젝트 시작할 때 보일러플레이트 생성

```bash
ollama run qwen2.5-coder:32b "Go로 Clean Architecture 구조의 프로젝트 보일러플레이트를 만들어줘.
디렉토리 구조와 주요 파일 코드를 포함해."
```

### 2. 에러 메시지 해석

```bash
echo "panic: runtime error: index out of range [5] with length 3" | \
  ollama run qwen2.5-coder:32b "이 Go 에러가 무슨 뜻이고 어떻게 해결해?"
```

### 3. SQL 쿼리 작성

```bash
ollama run qwen2.5-coder:32b "PostgreSQL에서 users 테이블과 orders 테이블을 JOIN해서
최근 30일 이내 주문한 사용자의 이름과 총 주문금액을 구하는 쿼리를 작성해줘"
```

### 4. 정규식 작성

```bash
ollama run qwen2.5-coder:32b "이메일 주소를 검증하는 정규식을 Python으로 작성해줘"
```

### 5. git diff 리뷰

```bash
git diff HEAD~1 | ollama run qwen2.5-coder:32b "이 변경사항을 리뷰해줘"
```

## 성능 최적화 팁

### 코드 생성 시 temperature 낮추기

```bash
# API로 호출할 때 temperature를 0.1~0.2로 설정하면 더 정확한 코드를 생성한다
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "...",
  "stream": false,
  "options": {"temperature": 0.1}
}'
```

### 긴 코드를 분석할 때 컨텍스트 확대

128GB RAM이므로 컨텍스트를 크게 잡아도 된다:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:32b",
  "prompt": "...(긴 코드)...",
  "stream": false,
  "options": {"num_ctx": 32768}
}'
```

## FAQ

### Q: 쿨러가 시끄럽게 돈다?

**정상이다.** M4 Max의 GPU가 95%+ 사용률로 추론을 수행하기 때문.
Activity Monitor에서 CPU 사용량은 낮고, GPU 사용량이 높은 것이 정상.

### Q: 응답이 너무 느리다?

- 모델이 처음 로드될 때 10~30초 걸린다 (웜업)
- 이후 응답은 빨라진다 (15~20 tokens/s)
- `ollama ps`로 모델이 메모리에 로드되어 있는지 확인

### Q: 메모리가 부족하다고 나온다?

```bash
# 사용하지 않는 모델을 메모리에서 내리기
curl http://localhost:11434/api/generate -d '{"model":"모델명","keep_alive":0}'
```

### Q: 모델이 한국어를 잘 못한다?

프롬프트에 "한국어로 답변해줘"를 추가하면 개선된다.
system 메시지로 지정하는 것이 더 효과적:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5-coder:32b",
  "messages": [
    {"role": "system", "content": "항상 한국어로 답변해. 코드 주석도 한국어로."},
    {"role": "user", "content": "..."}
  ],
  "stream": false
}'
```

### Q: 72b 모델과 32b 모델의 차이는?

| | 32B | 72B |
|--|-----|-----|
| 코드 정확도 | 우수 | 더 우수 |
| 설명/분석 | 좋음 | 매우 좋음 |
| 속도 | 16~20 t/s | 8~12 t/s |
| 메모리 | ~22GB | ~50GB |

코드만 짧게 생성할 때는 32B가 효율적이고,
복잡한 분석/설명이 필요할 때는 72B가 낫다.
