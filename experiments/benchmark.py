"""
로컬 LLM 코드 생성 벤치마크
- 여러 모델을 다양한 시나리오로 평가
- 결과를 experiments/results/ 에 JSON + Markdown 리포트로 저장
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime

BASE_URL = "http://localhost:11434"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── 평가할 모델 목록 ──

MODELS = [
    "qwen2.5-coder:32b",
    "qwen2.5-coder:32b-instruct-q8_0",
    "qwen2.5:72b",
    "deepseek-coder-v2",
]

# ── 테스트 시나리오 ──

SCENARIOS = [
    # === 카테고리 1: 알고리즘 구현 ===
    {
        "id": "algo-01",
        "category": "알고리즘",
        "name": "이진 탐색 트리 구현",
        "prompt": "Python으로 이진 탐색 트리(BST)를 구현해줘. insert, search, delete, inorder_traversal 메서드가 있어야 해. 코드만 작성해.",
        "checks": ["class", "def insert", "def search", "def delete"],
        "language": "python",
    },
    {
        "id": "algo-02",
        "category": "알고리즘",
        "name": "다익스트라 최단 경로",
        "prompt": "Python으로 다익스트라 알고리즘을 구현해줘. 인접 리스트로 그래프를 표현하고 heapq를 사용해. 코드만 작성해.",
        "checks": ["heapq", "def dijkstra", "distance"],
        "language": "python",
    },
    {
        "id": "algo-03",
        "category": "알고리즘",
        "name": "LRU 캐시 구현",
        "prompt": "Python으로 LRU Cache를 구현해줘. OrderedDict 없이 직접 더블 링크드 리스트와 해시맵으로 구현해. get, put 메서드. 코드만 작성해.",
        "checks": ["class", "def get", "def put"],
        "language": "python",
    },

    # === 카테고리 2: 웹 서버/API ===
    {
        "id": "web-01",
        "category": "웹/API",
        "name": "Go REST API 서버",
        "prompt": "Go언어로 net/http만 사용하여 TODO REST API를 만들어줘. GET /todos, POST /todos, DELETE /todos/{id} 엔드포인트. JSON 응답. 코드만 작성해.",
        "checks": ["func", "http.HandleFunc", "json", "DELETE"],
        "language": "go",
    },
    {
        "id": "web-02",
        "category": "웹/API",
        "name": "Python FastAPI CRUD",
        "prompt": "Python FastAPI로 사용자 CRUD API를 만들어줘. Pydantic 모델 사용, GET/POST/PUT/DELETE 엔드포인트. 인메모리 저장소. 코드만 작성해.",
        "checks": ["FastAPI", "class", "def create", "def delete"],
        "language": "python",
    },
    {
        "id": "web-03",
        "category": "웹/API",
        "name": "Go 미들웨어 체인",
        "prompt": "Go언어로 HTTP 미들웨어 체인을 구현해줘. 로깅 미들웨어, 인증 미들웨어, CORS 미들웨어를 만들고 체이닝할 수 있게 해줘. 코드만 작성해.",
        "checks": ["func", "http.Handler", "middleware"],
        "language": "go",
    },

    # === 카테고리 3: 디자인 패턴 ===
    {
        "id": "pattern-01",
        "category": "디자인패턴",
        "name": "Strategy 패턴",
        "prompt": "Python으로 Strategy 패턴을 구현해줘. 결제 시스템을 예시로 해서 CreditCard, PayPal, Bitcoin 결제 전략을 만들어. 코드만 작성해.",
        "checks": ["class", "CreditCard", "PayPal", "def pay"],
        "language": "python",
    },
    {
        "id": "pattern-02",
        "category": "디자인패턴",
        "name": "Go 인터페이스 + Factory",
        "prompt": "Go언어로 Factory 패턴을 구현해줘. Shape 인터페이스(Area, Perimeter 메서드)를 정의하고 Circle, Rectangle, Triangle 구현체를 만들어. NewShape 팩토리 함수도 포함. 코드만 작성해.",
        "checks": ["interface", "struct", "func New", "Circle", "Rectangle"],
        "language": "go",
    },

    # === 카테고리 4: 동시성/비동기 ===
    {
        "id": "concurrency-01",
        "category": "동시성",
        "name": "Go 워커 풀",
        "prompt": "Go언어로 Worker Pool 패턴을 구현해줘. goroutine과 channel을 사용해서 작업 큐를 처리하는 워커 풀을 만들어. 워커 수를 지정할 수 있어야 해. 코드만 작성해.",
        "checks": ["goroutine", "chan", "func worker", "go "],
        "language": "go",
    },
    {
        "id": "concurrency-02",
        "category": "동시성",
        "name": "Python 비동기 웹 크롤러",
        "prompt": "Python asyncio와 aiohttp를 사용해서 비동기 웹 크롤러를 만들어줘. URL 리스트를 받아서 동시에 요청하고 결과를 수집해. Semaphore로 동시 요청 수를 제한해. 코드만 작성해.",
        "checks": ["async", "await", "aiohttp", "Semaphore"],
        "language": "python",
    },

    # === 카테고리 5: 데이터 처리 ===
    {
        "id": "data-01",
        "category": "데이터처리",
        "name": "CSV 파서 구현",
        "prompt": "Python으로 CSV 파서를 직접 구현해줘. csv 모듈을 쓰지 말고 직접 파싱해. 따옴표 안의 쉼표, 줄바꿈도 처리할 수 있어야 해. 코드만 작성해.",
        "checks": ["def parse", "quote", "class"],
        "language": "python",
    },
    {
        "id": "data-02",
        "category": "데이터처리",
        "name": "Go JSON 스트림 파서",
        "prompt": "Go언어로 대용량 JSON 파일을 스트리밍 방식으로 파싱하는 코드를 작성해줘. encoding/json의 Decoder를 사용하고 메모리를 효율적으로 사용해. 코드만 작성해.",
        "checks": ["json.NewDecoder", "Decode", "func"],
        "language": "go",
    },

    # === 카테고리 6: 코드 분석 / 버그 수정 ===
    {
        "id": "debug-01",
        "category": "디버깅",
        "name": "메모리 누수 코드 분석",
        "prompt": """다음 Go 코드에서 메모리 누수가 발생하는 원인을 찾고 수정해줘. 한국어로 설명:

func process(ch chan string) {
    ticker := time.NewTicker(time.Second)
    for {
        select {
        case msg := <-ch:
            fmt.Println(msg)
        case <-ticker.C:
            fmt.Println("tick")
        }
    }
}

func main() {
    for i := 0; i < 100; i++ {
        ch := make(chan string)
        go process(ch)
        ch <- "hello"
    }
    time.Sleep(time.Minute)
}""",
        "checks": ["ticker.Stop", "goroutine"],
        "language": "go",
    },
    {
        "id": "debug-02",
        "category": "디버깅",
        "name": "레이스 컨디션 수정",
        "prompt": """다음 Python 코드의 레이스 컨디션을 찾아 수정해줘. 한국어로 설명:

import threading

counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(counter)  # 1000000이 아닌 다른 값이 나옴""",
        "checks": ["Lock", "lock", "threading"],
        "language": "python",
    },

    # === 카테고리 7: 프로젝트 설명 ===
    {
        "id": "explain-01",
        "category": "코드설명",
        "name": "복잡한 데코레이터 설명",
        "prompt": """다음 Python 코드가 어떻게 동작하는지 초보자도 이해할 수 있게 한국어로 설명해줘:

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator""",
        "checks": ["데코레이터", "재시도"],
        "language": "explanation",
    },
]


def get_available_models():
    """현재 설치된 모델 목록 조회"""
    try:
        resp = urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=10)
        data = json.loads(resp.read())
        return [m["name"] for m in data["models"]]
    except Exception as e:
        print(f"Ollama 서비스에 연결할 수 없습니다: {e}")
        sys.exit(1)


def api_call(model, prompt, num_ctx=8192, timeout=300):
    """Ollama API 호출"""
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": num_ctx},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def evaluate_response(response_text, checks):
    """응답 품질 평가"""
    score = 0
    check_results = []
    for check in checks:
        found = check.lower() in response_text.lower()
        check_results.append({"keyword": check, "found": found})
        if found:
            score += 1
    return {
        "score": score,
        "total": len(checks),
        "percentage": round(score / len(checks) * 100) if checks else 0,
        "checks": check_results,
    }


def run_benchmark(models, scenarios):
    """전체 벤치마크 실행"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": models,
            "scenario_count": len(scenarios),
        },
        "results": [],
        "summary": {},
    }

    total = len(models) * len(scenarios)
    current = 0

    for model in models:
        print(f"\n{'#'*60}")
        print(f"  모델: {model}")
        print(f"{'#'*60}")

        model_results = []

        for scenario in scenarios:
            current += 1
            sid = scenario["id"]
            print(f"\n  [{current}/{total}] {scenario['category']} > {scenario['name']}")

            try:
                start_time = time.time()
                raw = api_call(model, scenario["prompt"])
                elapsed = round(time.time() - start_time, 1)

                response = raw["response"]
                eval_count = raw.get("eval_count", 0)
                eval_duration = raw.get("eval_duration", 1) / 1e9
                tokens_per_sec = round(eval_count / eval_duration, 1) if eval_duration > 0 else 0

                evaluation = evaluate_response(response, scenario["checks"])

                result = {
                    "scenario_id": sid,
                    "category": scenario["category"],
                    "name": scenario["name"],
                    "language": scenario["language"],
                    "model": model,
                    "status": "success",
                    "elapsed_sec": elapsed,
                    "tokens": eval_count,
                    "tokens_per_sec": tokens_per_sec,
                    "response_length": len(response),
                    "evaluation": evaluation,
                    "response": response,
                }

                print(f"    시간: {elapsed}s | 토큰: {eval_count} | 속도: {tokens_per_sec} t/s | 점수: {evaluation['score']}/{evaluation['total']}")

            except Exception as e:
                result = {
                    "scenario_id": sid,
                    "category": scenario["category"],
                    "name": scenario["name"],
                    "language": scenario["language"],
                    "model": model,
                    "status": "error",
                    "error": str(e),
                }
                print(f"    ERROR: {e}")

            model_results.append(result)
            results["results"].append(result)

        # 모델별 요약
        successes = [r for r in model_results if r["status"] == "success"]
        if successes:
            avg_score = round(
                sum(r["evaluation"]["percentage"] for r in successes) / len(successes), 1
            )
            avg_speed = round(
                sum(r["tokens_per_sec"] for r in successes) / len(successes), 1
            )
            avg_time = round(
                sum(r["elapsed_sec"] for r in successes) / len(successes), 1
            )
            results["summary"][model] = {
                "avg_score": avg_score,
                "avg_speed_tps": avg_speed,
                "avg_time_sec": avg_time,
                "success_count": len(successes),
                "error_count": len(model_results) - len(successes),
            }

    # JSON 결과 저장
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 결과 저장: {json_path}")

    # Markdown 리포트 생성
    report_path = os.path.join(RESULTS_DIR, f"report_{timestamp}.md")
    generate_report(results, report_path)
    print(f"리포트 저장: {report_path}")

    return results


def generate_report(results, path):
    """Markdown 리포트 생성"""
    meta = results["metadata"]
    lines = [
        f"# 로컬 LLM 코드 생성 벤치마크 리포트",
        f"",
        f"- 실행 일시: {meta['date']}",
        f"- 평가 모델: {', '.join(meta['models'])}",
        f"- 시나리오 수: {meta['scenario_count']}개",
        f"",
        f"## 모델별 종합 점수",
        f"",
        f"| 모델 | 평균 점수 | 평균 속도 (t/s) | 평균 응답시간 | 성공/실패 |",
        f"|------|----------|----------------|-------------|----------|",
    ]

    for model, summary in results["summary"].items():
        lines.append(
            f"| {model} | {summary['avg_score']}% | {summary['avg_speed_tps']} | {summary['avg_time_sec']}s | {summary['success_count']}/{summary['error_count']} |"
        )

    lines += ["", "## 카테고리별 상세 결과", ""]

    # 카테고리별 그룹핑
    categories = {}
    for r in results["results"]:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, items in categories.items():
        lines += [f"### {cat}", ""]
        lines += [
            "| 시나리오 | 모델 | 점수 | 속도 (t/s) | 시간 |",
            "|---------|------|------|-----------|------|",
        ]
        for r in items:
            if r["status"] == "success":
                ev = r["evaluation"]
                lines.append(
                    f"| {r['name']} | {r['model']} | {ev['score']}/{ev['total']} ({ev['percentage']}%) | {r['tokens_per_sec']} | {r['elapsed_sec']}s |"
                )
            else:
                lines.append(f"| {r['name']} | {r['model']} | ERROR | - | - |")
        lines.append("")

    # 각 시나리오별 생성된 코드 첨부
    lines += ["## 생성된 코드 상세", ""]
    for r in results["results"]:
        if r["status"] == "success":
            lang = r["language"] if r["language"] != "explanation" else ""
            lines += [
                f"### {r['name']} ({r['model']})",
                f"",
                f"- 점수: {r['evaluation']['score']}/{r['evaluation']['total']}",
                f"- 키워드 체크: {', '.join(c['keyword'] + ('✓' if c['found'] else '✗') for c in r['evaluation']['checks'])}",
                f"- 응답 시간: {r['elapsed_sec']}s, 토큰: {r['tokens']}, 속도: {r['tokens_per_sec']} t/s",
                f"",
                f"```{lang}",
                r["response"][:2000],
                "```",
                "",
            ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    print("=" * 60)
    print("  로컬 LLM 코드 생성 벤치마크")
    print("=" * 60)

    available = get_available_models()
    print(f"\n설치된 모델: {available}")

    # 설치된 모델만 필터
    models_to_test = [m for m in MODELS if m in available]

    if not models_to_test:
        print("테스트할 모델이 없습니다. 모델을 먼저 설치하세요.")
        sys.exit(1)

    print(f"테스트 대상 모델: {models_to_test}")
    print(f"시나리오 수: {len(SCENARIOS)}개")
    print(f"총 테스트 수: {len(models_to_test) * len(SCENARIOS)}개")

    results = run_benchmark(models_to_test, SCENARIOS)

    # 최종 요약
    print(f"\n{'='*60}")
    print("  최종 결과 요약")
    print(f"{'='*60}")
    for model, summary in results["summary"].items():
        print(f"\n  {model}:")
        print(f"    평균 점수:     {summary['avg_score']}%")
        print(f"    평균 속도:     {summary['avg_speed_tps']} tokens/s")
        print(f"    평균 응답시간: {summary['avg_time_sec']}s")
