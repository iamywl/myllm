#!/bin/bash
# qwen2.5:72b 모델 설치 스크립트
# 사용법: ./install_72b.sh
#
# 47GB 다운로드가 필요합니다. 네트워크 상태에 따라 30~60분 소요됩니다.
# 설치 후 기존 32b 모델을 삭제하여 디스크 공간을 확보할 수 있습니다.

set -e

echo "============================================"
echo "  qwen2.5:72b 모델 설치"
echo "============================================"
echo ""
echo "  모델 크기: 47GB"
echo "  예상 소요: 30~60분 (네트워크 속도에 따라 다름)"
echo ""

# Ollama 서비스 확인
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[!] Ollama 서비스가 실행 중이 아닙니다."
    echo "    brew services start ollama"
    exit 1
fi

echo "[1/3] qwen2.5:72b 다운로드 시작..."
ollama pull qwen2.5:72b

echo ""
echo "[2/3] 설치 확인..."
ollama show qwen2.5:72b --modelfile | head -5

echo ""
echo "[3/3] 간단한 테스트..."
RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:72b",
  "prompt": "1+1=?",
  "stream": false
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['response'][:100])")

echo "  테스트 응답: $RESPONSE"

echo ""
echo "============================================"
echo "  설치 완료!"
echo "============================================"
echo ""
echo "  사용법:"
echo "    ollama run qwen2.5:72b \"질문\""
echo ""
echo "  기존 32b 모델 삭제 (선택):"
echo "    ollama rm qwen2.5-coder:32b"
echo "    ollama rm deepseek-coder-v2"
echo ""

ollama list
