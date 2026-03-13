"""Ollama 로컬 LLM 통합 테스트"""

import json
import subprocess
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
        func()
        print(f"  -> PASS")
        passed += 1
    except Exception as e:
        print(f"  -> FAIL: {e}")
        failed += 1


def api_call(endpoint, data):
    """Ollama API 호출 헬퍼"""
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ── 테스트 1: 서비스 상태 확인 ──


def test_service_running():
    resp = urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=10)
    data = json.loads(resp.read())
    models = [m["name"] for m in data["models"]]
    print(f"  설치된 모델: {models}")
    assert any("qwen2.5-coder" in m for m in models), "qwen2.5-coder 모델이 없습니다"


test("Ollama 서비스 및 모델 확인", test_service_running)


# ── 테스트 2: 한국어 설명 능력 ──


def test_korean_explanation():
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "Python 데코레이터가 뭔지 한 문장으로 설명해줘. 한국어로.",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답: {response[:200]}")
    assert len(response) > 10, "응답이 너무 짧습니다"
    # 한국어가 포함되어 있는지 확인
    assert any('\uAC00' <= c <= '\uD7A3' for c in response), "한국어 응답이 아닙니다"


test("한국어 설명 능력", test_korean_explanation)


# ── 테스트 3: Python 코드 생성 ──


def test_python_codegen():
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "Python으로 버블소트 함수를 작성해줘. 코드만 출력해.",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "def " in response, "Python 함수 정의가 없습니다"
    assert "sort" in response.lower(), "정렬 관련 코드가 없습니다"
    print(f"  코드 미리보기:\n{response[:300]}")


test("Python 코드 생성", test_python_codegen)


# ── 테스트 4: Go 코드 생성 ──


def test_go_codegen():
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": "Go언어로 두 숫자를 더하는 함수를 작성해. 코드만 출력해.",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답 길이: {len(response)}자")
    assert "func" in response, "Go 함수 정의가 없습니다"
    print(f"  코드 미리보기:\n{response[:300]}")


test("Go 코드 생성", test_go_codegen)


# ── 테스트 5: Chat API (멀티턴) ──


def test_chat_api():
    result = api_call("/api/chat", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "x = 10이라고 기억해줘"},
            {"role": "assistant", "content": "네, x = 10으로 기억하겠습니다."},
            {"role": "user", "content": "x의 값은?"},
        ],
        "stream": False,
    })
    response = result["message"]["content"]
    print(f"  응답: {response[:200]}")
    assert "10" in response, "멀티턴 컨텍스트를 유지하지 못합니다"


test("Chat API 멀티턴 대화", test_chat_api)


# ── 테스트 6: OpenAI 호환 API ──


def test_openai_compatible():
    result = api_call("/v1/chat/completions", {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "1+1=?"},
        ],
    })
    content = result["choices"][0]["message"]["content"]
    print(f"  응답: {content[:200]}")
    assert "2" in content, "올바른 답변이 아닙니다"


test("OpenAI 호환 API", test_openai_compatible)


# ── 테스트 7: 코드 분석 능력 ──


def test_code_analysis():
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    result = api_call("/api/generate", {
        "model": MODEL,
        "prompt": f"다음 코드의 동작 원리를 한국어로 설명해줘:\n{sample_code}",
        "stream": False,
    })
    response = result["response"]
    print(f"  응답: {response[:300]}")
    assert len(response) > 50, "설명이 너무 짧습니다"
    assert any('\uAC00' <= c <= '\uD7A3' for c in response), "한국어 응답이 아닙니다"


test("코드 분석/설명 능력", test_code_analysis)


# ── 결과 출력 ──

print(f"\n{'='*60}")
print(f"테스트 결과: {passed} passed, {failed} failed / 총 {passed + failed}개")
print(f"{'='*60}")

sys.exit(1 if failed > 0 else 0)
