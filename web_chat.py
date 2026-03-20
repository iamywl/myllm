#!/usr/bin/env python3
"""로컬 웹 챗봇 - Ollama 기반 대화형 웹 인터페이스"""

import json
import os
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from rag_store import RAGStore

MODEL = "devstral-2:123b"
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/chat"
HISTORY_DIR = "tmp/chat"
PORT = 60000

os.makedirs(HISTORY_DIR, exist_ok=True)

# 세션별 시스템 프롬프트
SESSION_PRESETS = {
    "Tech": {
        "system": (
            "너는 시니어 소프트웨어 엔지니어이자 기술 상담사다. "
            "사용자와 나눈 이전 대화 내용을 모두 기억하고 참조하며 답변한다. "
            "이전에 논의한 기술 결정, 코드, 아키텍처, 문제 해결 과정을 항상 맥락에 포함시켜라. "
            "이전 대화에서 언급된 내용이 현재 질문과 관련이 있다면 반드시 연결지어 설명해라. "
            "답변은 구체적이고 실용적으로, 코드 예시를 포함해서 해라."
        ),
        "label": "Tech (기술 상담)",
        "color": "#00d4aa",
        "use_rag": True,
    },
    "Mental": {
        "system": (
            "너는 따뜻하고 공감 능력이 뛰어난 심리 상담사다. "
            "사용자의 감정을 먼저 인정하고 공감한 뒤 조언해라. "
            "판단하지 말고 경청하는 자세로 대화해라. "
            "필요하다면 구체적인 대처 방법이나 사고 전환 기법을 제안해라. "
            "답변은 부드럽고 인간적인 톤으로 해라."
        ),
        "label": "Mental (심리 상담)",
        "color": "#e9a045",
        "use_rag": False,
    },
}
DEFAULT_SESSION = "Tech"


def measure_timeout():
    """GPU 사용량과 실측 추론 속도를 기반으로 타임아웃을 계산한다."""
    try:
        # 1) 모델 GPU 오프로드 비율 확인
        ps_req = urllib.request.Request(f"{OLLAMA_BASE}/api/ps")
        ps_resp = json.loads(urllib.request.urlopen(ps_req, timeout=10).read())
        models = ps_resp.get("models", [])
        gpu_ratio = 1.0
        for m in models:
            if m.get("model", "").startswith(MODEL.split(":")[0]):
                total = m.get("size", 1)
                vram = m.get("size_vram", total)
                gpu_ratio = vram / total
                break

        # 2) 짧은 요청으로 실제 tokens/sec 측정
        bench_req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=json.dumps({
                "model": MODEL, "prompt": "1+1=", "stream": False
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        bench_resp = json.loads(
            urllib.request.urlopen(bench_req, timeout=120).read()
        )
        eval_count = bench_resp.get("eval_count", 1)
        eval_ns = bench_resp.get("eval_duration", 1)
        prompt_count = bench_resp.get("prompt_eval_count", 1)
        prompt_ns = bench_resp.get("prompt_eval_duration", 1)

        gen_tps = eval_count / (eval_ns / 1e9)       # 생성 속도
        prompt_tps = prompt_count / (prompt_ns / 1e9)  # 프롬프트 처리 속도

        # 3) 타임아웃 계산
        #    - 입력: 최대 컨텍스트 256K 토큰 (긴 대화 누적)
        #    - 출력: 최대 32K 토큰 응답
        max_ctx = 262144
        max_response = 32768
        prompt_time = max_ctx / prompt_tps
        gen_time = max_response / gen_tps
        timeout_sec = int((prompt_time + gen_time) * 1.2)

        print(f"  GPU 비율: {gpu_ratio*100:.0f}%")
        print(f"  프롬프트 처리: {prompt_tps:.1f} tokens/sec")
        print(f"  생성 속도: {gen_tps:.1f} tokens/sec")
        print(f"  최악 시나리오: 입력 {max_ctx//1024}K + 출력 {max_response//1024}K 토큰")
        print(f"  타임아웃: {timeout_sec}초 ({timeout_sec//60}분)")
        return timeout_sec

    except Exception as e:
        fallback = 3600
        print(f"  속도 측정 실패({e}), 기본값 {fallback}초 사용")
        return fallback


# 서버 시작 시 한번 측정
print("추론 속도 측정 중...")
REQUEST_TIMEOUT = measure_timeout()

# RAG 저장소 초기화
print("RAG 저장소 초기화 중...")
rag_store = RAGStore()

# 기존 대화 기록 자동 마이그레이션
if os.path.exists(HISTORY_DIR):
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        session = fname[:-5]
        filepath = os.path.join(HISTORY_DIR, fname)
        with open(filepath) as _f:
            msgs = json.load(_f)
        new_count = rag_store.migrate_from_messages(session, msgs)
        if new_count > 0:
            print(f"  [{session}] {new_count}턴 인덱싱 완료")
print("RAG 준비 완료.")


# 토큰 수 추정 (한글 ~2토큰/자, 영어 ~0.25토큰/자)
def estimate_tokens(messages):
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars * 1.5)


RECENT_TURNS = 6        # 최근 대화 유지 수 (user+assistant 쌍)
RAG_TOP_K = 10           # 검색 결과 수
TOKEN_LIMIT = 200000     # RAG 전환 기준 토큰 수


def build_rag_prompt(session, user_msg, full_messages):
    """RAG 검색 결과 + 최근 대화로 프롬프트를 조립한다."""
    prompt_messages = []

    # 1) 시스템 프롬프트
    preset = SESSION_PRESETS.get(session, {})
    system = preset.get("system", "")
    if system:
        prompt_messages.append({"role": "system", "content": system})

    # 2) RAG 검색 → 관련 과거 대화 삽입
    relevant = rag_store.search(user_msg, session, top_k=RAG_TOP_K)
    if relevant:
        context_parts = []
        for r in relevant:
            context_parts.append(f"[대화 #{r['turn_index']}] {r['content']}")
        context = "\n\n---\n\n".join(context_parts)
        prompt_messages.append({
            "role": "system",
            "content": f"다음은 이전 대화에서 현재 질문과 관련된 부분이다. 참고하여 답변하라:\n\n{context}"
        })

    # 3) 최근 대화 (직전 맥락 유지)
    non_system = [m for m in full_messages if m["role"] != "system"]
    recent = non_system[-(RECENT_TURNS * 2):]
    prompt_messages.extend(recent)

    return prompt_messages

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MyLLM Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #16213e;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #0f3460;
  }
  header h1 { font-size: 18px; color: #e94560; }
  .header-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .session-bar {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .session-bar select, .session-bar input, .session-bar button {
    background: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 13px;
  }
  .session-bar button {
    background: #e94560;
    border: none;
    cursor: pointer;
    font-weight: bold;
  }
  .session-bar button:hover { background: #c73e54; }
  .session-indicator {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 10px;
    font-weight: bold;
  }
  .ctrl-bar { display: flex; gap: 6px; }
  .ctrl-bar button {
    background: #333;
    color: #ccc;
    border: 1px solid #555;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
  }
  .ctrl-bar button:hover { background: #555; }
  .ctrl-bar button.danger { border-color: #e94560; color: #e94560; }
  .ctrl-bar button.danger:hover { background: #e94560; color: white; }
  #chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 14px;
  }
  .msg.user {
    align-self: flex-end;
    background: #0f3460;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start;
    background: #16213e;
    border-bottom-left-radius: 4px;
    border: 1px solid #0f3460;
  }
  .msg.assistant code {
    background: #1a1a2e;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 13px;
  }
  .msg.assistant pre {
    background: #1a1a2e;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .msg.assistant pre code {
    background: none;
    padding: 0;
  }
  .typing {
    align-self: flex-start;
    color: #888;
    font-style: italic;
    padding: 8px 16px;
  }
  #input-area {
    padding: 16px 20px;
    background: #16213e;
    border-top: 1px solid #0f3460;
    display: flex;
    gap: 10px;
  }
  #input-area textarea {
    flex: 1;
    background: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    min-height: 48px;
    max-height: 150px;
    outline: none;
  }
  #input-area textarea:focus { border-color: #e94560; }
  #send-btn {
    background: #e94560;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0 20px;
    font-size: 15px;
    cursor: pointer;
    font-weight: bold;
  }
  #send-btn:hover { background: #c73e54; }
  #send-btn:disabled { background: #555; cursor: not-allowed; }
  #chat-area::-webkit-scrollbar { width: 6px; }
  #chat-area::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div class="header-row">
    <h1>MyLLM Chat</h1>
    <span class="session-indicator" id="session-indicator">Tech</span>
  </div>
  <div class="header-row">
    <div class="session-bar">
      <select id="session-select"></select>
      <input id="new-session" type="text" placeholder="새 세션명" size="10">
      <button onclick="createSession()">새 세션</button>
    </div>
    <div class="ctrl-bar">
      <button onclick="unloadModel()">모델 내리기</button>
      <button class="danger" onclick="stopServer()">서버 종료</button>
    </div>
  </div>
</header>

<div id="chat-area"></div>

<div id="input-area">
  <textarea id="input" placeholder="메시지를 입력하세요... (Shift+Enter: 줄바꿈, Enter: 전송)" rows="1"></textarea>
  <button id="send-btn" onclick="sendMessage()">전송</button>
</div>

<script>
const chatArea = document.getElementById('chat-area');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const sessionSelect = document.getElementById('session-select');
const indicator = document.getElementById('session-indicator');

const PRESETS = {
  Tech:   { label: 'Tech (기술 상담)', color: '#00d4aa' },
  Mental: { label: 'Mental (심리 상담)', color: '#e9a045' },
};

let currentSession = 'Tech';
let sending = false;

function updateIndicator() {
  const p = PRESETS[currentSession];
  indicator.textContent = p ? p.label : currentSession;
  indicator.style.background = (p ? p.color : '#e94560') + '22';
  indicator.style.color = p ? p.color : '#e94560';
}

async function unloadModel() {
  if (!confirm('모델을 메모리에서 내립니다. 다음 요청 시 자동 재로드됩니다.')) return;
  try {
    await fetch('/api/unload', { method: 'POST' });
    alert('모델이 메모리에서 해제되었습니다.');
  } catch(e) { alert('오류: ' + e); }
}

async function stopServer() {
  if (!confirm('웹 챗봇 서버를 종료합니다.')) return;
  try {
    await fetch('/api/shutdown', { method: 'POST' });
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#888;font-size:20px;">서버가 종료되었습니다. 브라우저를 닫아주세요.</div>';
  } catch(e) {}
}

// 자동 높이 조절
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 150) + 'px';
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = content;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'typing';
  div.id = 'typing';
  div.textContent = '생각하는 중...';
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || sending) return;

  sending = true;
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  addMessage('user', text);
  showTyping();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session: currentSession, message: text })
    });
    const data = await resp.json();
    removeTyping();

    if (data.error) {
      addMessage('assistant', '오류: ' + data.error);
    } else {
      addMessage('assistant', data.reply);
    }
  } catch (err) {
    removeTyping();
    addMessage('assistant', '연결 오류: Ollama가 실행 중인지 확인하세요.');
  }

  sending = false;
  sendBtn.disabled = false;
  input.focus();
}

async function loadSessions() {
  const resp = await fetch('/api/sessions');
  const sessions = await resp.json();
  sessionSelect.innerHTML = '';
  sessions.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    if (s === currentSession) opt.selected = true;
    sessionSelect.appendChild(opt);
  });
}

sessionSelect.addEventListener('change', async () => {
  currentSession = sessionSelect.value;
  updateIndicator();
  await loadHistory();
});

async function loadHistory() {
  chatArea.innerHTML = '';
  const resp = await fetch('/api/history?session=' + encodeURIComponent(currentSession));
  const messages = await resp.json();
  messages.forEach(m => addMessage(m.role, m.content));
}

function createSession() {
  const name = document.getElementById('new-session').value.trim();
  if (!name) return;
  currentSession = name;
  document.getElementById('new-session').value = '';
  chatArea.innerHTML = '';
  updateIndicator();
  loadSessions();
}

// 초기화
updateIndicator();
loadSessions().then(() => loadHistory());
input.focus();
</script>
</body>
</html>"""


class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self._respond(200, 'text/html', HTML.encode())
        elif self.path == '/api/sessions':
            sessions = list(SESSION_PRESETS.keys())
            if os.path.exists(HISTORY_DIR):
                for f in sorted(os.listdir(HISTORY_DIR)):
                    if f.endswith('.json'):
                        name = f[:-5]
                        if name not in sessions:
                            sessions.append(name)
            self._respond(200, 'application/json',
                          json.dumps(sessions).encode())
        elif self.path.startswith('/api/history'):
            session = self.path.split('session=')[-1] if 'session=' in self.path else 'default'
            session = urllib.request.unquote(session)
            history_file = os.path.join(HISTORY_DIR, f"{session}.json")
            messages = []
            if os.path.exists(history_file):
                with open(history_file) as f:
                    messages = json.load(f)
            self._respond(200, 'application/json',
                          json.dumps(messages).encode())
        else:
            self._respond(404, 'text/plain', b'Not Found')

    def do_POST(self):
        if self.path == '/api/unload':
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_BASE}/api/generate",
                    data=json.dumps({
                        "model": MODEL, "keep_alive": 0
                    }).encode(),
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=30)
                self._respond(200, 'application/json',
                              json.dumps({"ok": True}).encode())
            except Exception as e:
                self._respond(500, 'application/json',
                              json.dumps({"error": str(e)}).encode())
            return

        if self.path == '/api/shutdown':
            self._respond(200, 'application/json',
                          json.dumps({"ok": True}).encode())
            print("\n웹 챗봇 서버 종료 요청됨.")
            import threading
            threading.Thread(target=self.server.shutdown).start()
            return

        if self.path != '/api/chat':
            self._respond(404, 'text/plain', b'Not Found')
            return

        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))
        session = body.get('session', DEFAULT_SESSION)
        user_msg = body.get('message', '')

        history_file = os.path.join(HISTORY_DIR, f"{session}.json")

        # 전체 대화 기록 로드 (원본)
        if os.path.exists(history_file):
            with open(history_file) as f:
                full_messages = json.load(f)
        else:
            full_messages = []

        # 시스템 프롬프트 삽입 (첫 대화 시)
        preset = SESSION_PRESETS.get(session, {})
        if preset and not any(m['role'] == 'system' for m in full_messages):
            full_messages.insert(0, {'role': 'system', 'content': preset['system']})

        full_messages.append({'role': 'user', 'content': user_msg})

        # RAG 사용 여부 결정
        use_rag = preset.get("use_rag", False)
        if not use_rag and estimate_tokens(full_messages) >= TOKEN_LIMIT:
            use_rag = True  # 토큰 한도 초과 시 자동 전환

        # Ollama에 보낼 프롬프트 조립
        if use_rag:
            send_messages = build_rag_prompt(session, user_msg, full_messages)
        else:
            send_messages = full_messages

        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({
                    'model': MODEL,
                    'messages': send_messages,
                    'stream': False
                }).encode(),
                headers={'Content-Type': 'application/json'}
            )
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
            result = json.loads(resp.read())
            reply = result['message']['content']

            # 전체 기록에 저장 (원본 유지)
            full_messages.append({'role': 'assistant', 'content': reply})
            with open(history_file, 'w') as f:
                json.dump(full_messages, f, ensure_ascii=False, indent=2)

            # RAG 인덱싱: 새 턴을 벡터 DB에 추가
            try:
                turn_index = len([m for m in full_messages if m['role'] == 'user'])
                rag_store.add_pair(session, user_msg, reply, turn_index)
                rag_store.set_indexed_turns(session, turn_index)
            except Exception:
                pass  # 인덱싱 실패해도 응답은 정상 반환

            self._respond(200, 'application/json',
                          json.dumps({'reply': reply}).encode())
        except Exception as e:
            self._respond(500, 'application/json',
                          json.dumps({'error': str(e)}).encode())

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), ChatHandler)
    print(f"MyLLM 웹 챗봇 시작!")
    print(f"  브라우저에서 열기: http://localhost:{PORT}")
    print(f"  모델: {MODEL}")
    print(f"  대화 기록: {HISTORY_DIR}/")
    print(f"  종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료됨.")
        server.server_close()
