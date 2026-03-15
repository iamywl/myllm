#!/bin/bash
# Devstral-2 123B 설치 스크립트
# - 기존 모델 전부 삭제 후 Devstral-2:123b (Q4_K_M, 75GB) 설치
# - SWE-bench Verified 72.2% (오픈소스 코딩 모델 1위)
# - 256K 컨텍스트 윈도우
#
# 사용법: chmod +x install_devstral2.sh && ./install_devstral2.sh

set -e

echo "============================================"
echo "  Devstral-2 123B 설치"
echo "============================================"
echo ""
echo "  모델: devstral-2:123b (Q4_K_M)"
echo "  크기: ~75GB 다운로드"
echo "  예상: 네트워크에 따라 30~90분"
echo ""

# ── 1단계: Ollama 서비스 확인 ──

echo "[1/4] Ollama 서비스 확인..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  [!] Ollama가 실행 중이 아닙니다. 시작합니다..."
    brew services start ollama
    sleep 3
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  [ERROR] Ollama를 시작할 수 없습니다."
        exit 1
    fi
fi
echo "  OK"
echo ""

# ── 2단계: 기존 모델 전부 삭제 ──

echo "[2/4] 기존 모델 삭제..."
MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')

if [ -z "$MODELS" ]; then
    echo "  삭제할 모델 없음"
else
    for model in $MODELS; do
        echo "  삭제: $model"
        ollama rm "$model" 2>/dev/null || true
    done
fi

echo ""
echo "  디스크 확인:"
df -h / | awk 'NR==2{printf "  사용: %s / 전체: %s / 여유: %s\n", $3, $2, $4}'
echo ""

# ── 3단계: Devstral-2 123B 다운로드 ──

echo "[3/4] devstral-2:123b 다운로드 시작..."
echo "  (75GB - 시간이 걸립니다. Ctrl+C로 중단 가능, 재실행 시 이어받기)"
echo ""

START=$(date +%s)
ollama pull devstral-2:123b
END=$(date +%s)

ELAPSED=$(( END - START ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))
echo ""
echo "  다운로드 완료: ${MINUTES}분 ${SECONDS}초"
echo ""

# ── 4단계: 설치 확인 및 테스트 ──

echo "[4/4] 설치 확인 및 테스트..."
echo ""
echo "  설치된 모델:"
ollama list
echo ""

echo "  간단한 코딩 테스트..."
RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "devstral-2:123b",
  "prompt": "Write a Python function that checks if a string is a palindrome. Code only.",
  "stream": false
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['response'][:300])" 2>/dev/null)

echo "  응답:"
echo "  $RESPONSE"

echo ""
echo "============================================"
echo "  설치 완료!"
echo "============================================"
echo ""
echo "  모델:    devstral-2:123b (123B params, Q4_K_M)"
echo "  메모리:  ~75GB 사용 예상"
echo "  컨텍스트: 256K 토큰"
echo ""
echo "  사용법:"
echo "    ollama run devstral-2:123b \"코드 작성해줘\""
echo ""
echo "  벤치마크:"
echo "    python3 benchmark.py"
echo ""
