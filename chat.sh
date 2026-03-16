#!/bin/bash
# 로컬 LLM 채팅 스크립트 - 대화 맥락을 기억합니다
# 사용법: ./chat.sh [세션명]
# 예시: ./chat.sh 상담

MODEL="devstral-2:123b"
SESSION="${1:-default}"
HISTORY_DIR="tmp/chat"
HISTORY_FILE="$HISTORY_DIR/${SESSION}.json"

mkdir -p "$HISTORY_DIR"

# 새 세션이면 초기화
if [ ! -f "$HISTORY_FILE" ]; then
    echo '[]' > "$HISTORY_FILE"
    echo "새 세션 시작: $SESSION"
else
    TURN_COUNT=$(python3 -c "import json; msgs=json.load(open('$HISTORY_FILE')); print(len([m for m in msgs if m['role']=='user']))")
    echo "기존 세션 이어서 진행: $SESSION (이전 대화 ${TURN_COUNT}턴)"
fi

echo "모델: $MODEL"
echo "대화 기록: $HISTORY_FILE"
echo "종료: quit 또는 Ctrl+C"
echo "---"

while true; do
    # 사용자 입력 받기
    echo ""
    printf "나> "
    INPUT=""
    while IFS= read -r line; do
        [ -z "$line" ] && break
        [ -n "$INPUT" ] && INPUT="$INPUT\n"
        INPUT="$INPUT$line"
    done

    [ -z "$INPUT" ] && continue
    [ "$INPUT" = "quit" ] && echo "세션 저장 완료: $HISTORY_FILE" && break

    # 파일 내용 읽기 (@파일경로)
    if [[ "$INPUT" == @* ]]; then
        FILEPATH="${INPUT#@}"
        FILEPATH=$(echo "$FILEPATH" | xargs)  # trim
        if [ -f "$FILEPATH" ]; then
            INPUT=$(cat "$FILEPATH")
            echo "(파일 로드: $FILEPATH, $(wc -c < "$FILEPATH" | xargs)bytes)"
        else
            echo "파일을 찾을 수 없습니다: $FILEPATH"
            continue
        fi
    fi

    # 대화 기록에 사용자 메시지 추가 & API 호출
    RESPONSE=$(python3 -c "
import json, sys, urllib.request

history_file = '$HISTORY_FILE'
user_input = '''$INPUT'''

with open(history_file) as f:
    messages = json.load(f)

messages.append({'role': 'user', 'content': user_input})

req = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=json.dumps({
        'model': '$MODEL',
        'messages': messages,
        'stream': False
    }).encode(),
    headers={'Content-Type': 'application/json'}
)

resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
assistant_msg = result['message']['content']

messages.append({'role': 'assistant', 'content': assistant_msg})

with open(history_file, 'w') as f:
    json.dump(messages, f, ensure_ascii=False, indent=2)

print(assistant_msg)
" 2>&1)

    echo ""
    echo "AI> $RESPONSE"
done
