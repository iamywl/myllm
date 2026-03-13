"""Ollama 심화 테스트 - 대용량 메모리 활용, 복잡한 코드 생성, 성능 측정"""

import json
import time
import sys
import urllib.request

BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5-coder:32b"

passed = 0
failed = 0


def test(name, func):
    global passed, failed
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        start = time.time()
        func()
        elapsed = time.time() - start
        print(f"  -> PASS ({elapsed:.1f}s)")
        passed += 1
    except Exception as e:
        print(f"  -> FAIL: {e}")
        failed += 1


def api_call(endpoint, data, timeout=180):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── 테스트 1: 대용량 컨텍스트 (16K) ──


def test_large_context():
    """128GB RAM을 활용하여 큰 컨텍스트 윈도우로 코드 분석"""
    large_code = """
class UserService:
    def __init__(self, db, cache, logger):
        self.db = db
        self.cache = cache
        self.logger = logger

    def get_user(self, user_id):
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            self.logger.info(f"Cache hit for user {user_id}")
            return cached
        user = self.db.query("SELECT * FROM users WHERE id = %s", user_id)
        if user:
            self.cache.set(f"user:{user_id}", user, ttl=300)
        return user

    def create_user(self, name, email):
        if self.db.query("SELECT id FROM users WHERE email = %s", email):
            raise ValueError("Email already exists")
        user_id = self.db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)", name, email
        )
        self.logger.info(f"Created user {user_id}")
        return {"id": user_id, "name": name, "email": email}

    def update_user(self, user_id, **kwargs):
        user = self.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        sets = ", ".join(f"{k} = %s" for k in kwargs)
        self.db.execute(f"UPDATE users SET {sets} WHERE id = %s", *kwargs.values(), user_id)
        self.cache.delete(f"user:{user_id}")
        self.logger.info(f"Updated user {user_id}: {kwargs}")
        return self.get_user(user_id)

    def delete_user(self, user_id):
        self.db.execute("DELETE FROM users WHERE id = %s", user_id)
        self.cache.delete(f"user:{user_id}")
        self.logger.info(f"Deleted user {user_id}")

    def list_users(self, page=1, per_page=20):
        offset = (page - 1) * per_page
        users = self.db.query(
            "SELECT * FROM users ORDER BY id LIMIT %s OFFSET %s", per_page, offset
        )
        total = self.db.query("SELECT COUNT(*) FROM users")[0]
        return {"users": users, "total": total, "page": page, "per_page": per_page}
"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": f"다음 Python 클래스를 분석하고, 1) 각 메서드의 역할, 2) 사용된 디자인 패턴, 3) 잠재적 보안 문제를 한국어로 설명해줘:\n\n{large_code}",
        "stream": False,
        "options": {"num_ctx": 16384},
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    print(f"  미리보기: {response[:300]}...")
    assert len(response) > 200, "분석이 너무 짧습니다"
    assert "get_user" in response or "사용자" in response, "메서드 분석이 없습니다"


test("대용량 컨텍스트 코드 분석 (16K ctx)", test_large_context)


# ── 테스트 2: 복잡한 Python 코드 생성 ──


def test_complex_python():
    """디자인 패턴을 포함한 복잡한 코드 생성"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "Python으로 Observer 패턴을 구현해줘. EventEmitter 클래스를 만들고, on/emit/off 메서드가 있어야 해. 타입 힌트를 포함하고 사용 예제도 작성해줘. 코드만 출력해.",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "class" in response, "클래스 정의가 없습니다"
    assert "def on" in response or "def emit" in response, "필수 메서드가 없습니다"
    print(f"  미리보기:\n{response[:400]}")


test("복잡한 Python 코드 생성 (Observer 패턴)", test_complex_python)


# ── 테스트 3: Go 프로젝트 코드 생성 ──


def test_complex_go():
    """구조체, 인터페이스를 포함한 Go 코드 생성"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "Go언어로 간단한 인메모리 키-값 저장소를 만들어줘. interface로 Store를 정의하고 Get/Set/Delete 메서드가 있어야 해. sync.RWMutex로 동시성을 처리해줘. 코드만 출력해.",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "interface" in response or "Interface" in response or "struct" in response, "인터페이스/구조체가 없습니다"
    assert "Mutex" in response or "mutex" in response, "동시성 처리가 없습니다"
    print(f"  미리보기:\n{response[:400]}")


test("복잡한 Go 코드 생성 (KV Store)", test_complex_go)


# ── 테스트 4: 코드 리팩토링 능력 ──


def test_refactoring():
    bad_code = """
def process(data):
    result = []
    for item in data:
        if item['type'] == 'A':
            if item['value'] > 10:
                if item['active'] == True:
                    result.append(item['name'].upper())
                else:
                    result.append(item['name'].lower())
            else:
                result.append(item['name'])
        elif item['type'] == 'B':
            if item['value'] > 20:
                result.append(item['name'] + '_B')
            else:
                result.append(item['name'])
    return result
"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": f"다음 코드를 리팩토링해줘. 가독성을 높이고 early return 패턴을 사용해. 코드와 간단한 설명을 한국어로:\n\n{bad_code}",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "def" in response, "리팩토링된 함수가 없습니다"
    assert len(response) > 100, "리팩토링 결과가 너무 짧습니다"
    print(f"  미리보기:\n{response[:400]}")


test("코드 리팩토링 능력", test_refactoring)


# ── 테스트 5: 버그 찾기 ──


def test_bug_detection():
    buggy_code = """
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": f"이 코드에 버그가 있어. 찾아서 수정해주고 이유를 한국어로 설명해줘:\n\n{buggy_code}",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "mid" in response, "버그 위치를 지적하지 않았습니다"
    assert any('\uAC00' <= c <= '\uD7A3' for c in response), "한국어 설명이 없습니다"
    print(f"  미리보기:\n{response[:400]}")


test("버그 탐지 및 수정", test_bug_detection)


# ── 테스트 6: 성능 측정 (응답 속도) ──


def test_response_speed():
    """짧은 질문에 대한 응답 속도 측정"""
    start = time.time()
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "print('hello')",
        "stream": False,
    })
    elapsed = time.time() - start
    response = result["response"]

    tokens = result.get("eval_count", 0)
    eval_duration = result.get("eval_duration", 1) / 1e9  # ns -> s
    tokens_per_sec = tokens / eval_duration if eval_duration > 0 else 0

    print(f"  응답 시간: {elapsed:.1f}s")
    print(f"  토큰 수: {tokens}")
    print(f"  생성 속도: {tokens_per_sec:.1f} tokens/s")
    print(f"  응답: {response[:100]}")
    assert elapsed < 60, "응답이 60초를 초과했습니다"


test("응답 속도 측정", test_response_speed)


# ── 결과 출력 ──

print(f"\n{'='*60}")
print(f"심화 테스트 결과: {passed} passed, {failed} failed / 총 {passed + failed}개")
print(f"{'='*60}")

sys.exit(1 if failed > 0 else 0)
