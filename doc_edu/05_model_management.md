# 05. 모델 관리

## 현재 설치된 모델 확인

```bash
ollama list
```

## 모델 상세 정보

```bash
ollama show qwen2.5-coder:32b
```

파라미터 수, 양자화 방식, 컨텍스트 길이 등을 확인할 수 있다.

## 새 모델 설치

```bash
# 코딩 특화 (현재 사용 중)
ollama pull qwen2.5-coder:32b

# 범용 대형 모델 (47GB, 128GB RAM에서 구동 가능)
ollama pull qwen2.5:72b

# 경량 코딩 모델
ollama pull deepseek-coder-v2
```

72b 모델 설치 전용 스크립트도 있다:
```bash
./install_72b.sh
```

## 모델 삭제

```bash
# 특정 모델 삭제
ollama rm qwen2.5-coder:32b
ollama rm deepseek-coder-v2

# 삭제해도 언제든 다시 받을 수 있음
ollama pull qwen2.5-coder:32b
```

## 메모리에서 모델 언로드

모델을 삭제하지 않고 메모리에서만 내리고 싶을 때:

```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5-coder:32b","keep_alive":0}'
```

## 현재 메모리에 로드된 모델 확인

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool
```

## 128GB RAM에서 사용 가능한 모델 크기 가이드

| 모델 크기 | 디스크 | 메모리 사용 | 동시 로드 |
|----------|--------|-----------|----------|
| 7~8B (Q4) | 4~5GB | ~6GB | 여러 개 가능 |
| 14~16B (Q4) | 8~10GB | ~12GB | 여러 개 가능 |
| 32B (Q4) | 19GB | ~22GB | 2~3개 가능 |
| 72B (Q4) | 47GB | ~50GB | 1~2개 가능 |
| 32B (Q8) | 34GB | ~36GB | 2개 가능 |

> M4 Max 128GB에서는 72B 모델도 전체 GPU 오프로드가 되어 속도 저하 없이 동작한다.

## 서비스 관리

```bash
brew services start ollama    # 시작 (부팅 시 자동 시작)
brew services stop ollama     # 중지
brew services restart ollama  # 재시작
brew services info ollama     # 상태 확인
```

## 디스크 사용량 확인

모델은 `~/.ollama/models/`에 저장된다.

```bash
du -sh ~/.ollama/models/
```
