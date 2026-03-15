"""
로컬 LLM 종합 벤치마크
- 토큰 생성 속도 (tokens/s)
- TTFT (Time To First Token)
- GPU/CPU/메모리 사용량
- 토큰당 비용 추정
- 다양한 프롬프트 유형별 성능
"""

import json
import time
import subprocess
import urllib.request
import sys
import os

BASE_URL = "http://localhost:11434"
MODEL = "devstral-2:123b"

# ── 시스템 정보 수집 ──

def get_system_info():
    print("=" * 70)
    print("  시스템 정보")
    print("=" * 70)

    # 칩 정보
    hw = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True,
    )
    chip = hw.stdout.strip() or "Apple Silicon"

    # 실제 칩 이름
    chip_info = subprocess.run(
        ["system_profiler", "SPHardwareDataType"],
        capture_output=True, text=True,
    )
    for line in chip_info.stdout.splitlines():
        if "Chip" in line:
            chip = line.split(":")[-1].strip()
            break

    # 메모리
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    total_mem_gb = int(mem.stdout.strip()) / (1024 ** 3)

    # CPU 코어
    cores = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True)

    print(f"  칩: {chip}")
    print(f"  메모리: {total_mem_gb:.0f} GB")
    print(f"  CPU 코어: {cores.stdout.strip()}")
    print(f"  모델: {MODEL}")
    print(f"  OS: macOS {subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True).stdout.strip()}")
    print()
    return chip, total_mem_gb


# ── 리소스 모니터링 ──

def get_memory_usage():
    """Ollama 프로세스의 메모리 사용량 (MB)"""
    try:
        ps = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True,
        )
        total_rss = 0
        for line in ps.stdout.splitlines():
            if "ollama" in line.lower() and "grep" not in line:
                parts = line.split()
                if len(parts) >= 6:
                    total_rss += int(parts[5])  # RSS in KB
        return total_rss / 1024  # MB
    except Exception:
        return 0


def get_gpu_power():
    """Apple Silicon GPU 전력 사용량 추정 (powermetrics 없이)"""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", "1", "-n", "1"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "GPU Power" in line:
                return line.strip()
    except Exception:
        pass
    return None


def get_cpu_usage():
    """현재 CPU 사용률"""
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "%cpu,comm"],
            capture_output=True, text=True,
        )
        total = 0.0
        for line in result.stdout.splitlines():
            if "ollama" in line.lower():
                parts = line.strip().split()
                if parts:
                    try:
                        total += float(parts[0])
                    except ValueError:
                        pass
        return total
    except Exception:
        return 0.0


# ── API 호출 (스트리밍 / 논스트리밍) ──

def api_call_detailed(prompt, stream=False, num_ctx=65536, timeout=600):
    """상세 메트릭 포함 API 호출"""
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {"num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )

    if not stream:
        wall_start = time.time()
        mem_before = get_memory_usage()
        cpu_before = get_cpu_usage()

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())

        wall_end = time.time()
        mem_after = get_memory_usage()
        cpu_after = get_cpu_usage()

        # Ollama 내부 메트릭 (나노초)
        total_duration = result.get("total_duration", 0) / 1e9
        load_duration = result.get("load_duration", 0) / 1e9
        prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
        eval_duration = result.get("eval_duration", 0) / 1e9
        prompt_eval_count = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)

        tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
        prompt_tokens_per_sec = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0

        return {
            "response": result["response"],
            "wall_time": wall_end - wall_start,
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_duration": eval_duration,
            "prompt_tokens": prompt_eval_count,
            "eval_tokens": eval_count,
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens_per_sec": prompt_tokens_per_sec,
            "ttft": load_duration + prompt_eval_duration,  # 대략적 TTFT
            "mem_before_mb": mem_before,
            "mem_after_mb": mem_after,
            "cpu_usage": max(cpu_before, cpu_after),
        }
    else:
        # 스트리밍 - TTFT 측정
        wall_start = time.time()
        first_token_time = None
        tokens = []

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk
                if chunk == b"\n" and buffer.strip():
                    try:
                        obj = json.loads(buffer)
                        if obj.get("response"):
                            if first_token_time is None:
                                first_token_time = time.time()
                            tokens.append(obj["response"])
                        if obj.get("done"):
                            final = obj
                            break
                    except json.JSONDecodeError:
                        pass
                    buffer = b""

        wall_end = time.time()
        ttft = (first_token_time - wall_start) if first_token_time else 0

        eval_count = final.get("eval_count", len(tokens))
        eval_duration = final.get("eval_duration", 0) / 1e9
        tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0

        return {
            "response": "".join(tokens),
            "wall_time": wall_end - wall_start,
            "ttft": ttft,
            "eval_tokens": eval_count,
            "tokens_per_sec": tokens_per_sec,
        }


# ── 벤치마크 실행 ──

def print_metrics(label, m):
    print(f"\n  [{label}]")
    print(f"  총 소요 시간:       {m['wall_time']:.2f}s")
    if "ttft" in m:
        print(f"  TTFT (첫 토큰):     {m['ttft']:.2f}s")
    if "prompt_tokens" in m:
        print(f"  입력 토큰:          {m['prompt_tokens']}")
        print(f"  입력 처리 속도:     {m.get('prompt_tokens_per_sec', 0):.1f} tokens/s")
    print(f"  출력 토큰:          {m['eval_tokens']}")
    print(f"  생성 속도:          {m['tokens_per_sec']:.1f} tokens/s")
    if "mem_after_mb" in m and m["mem_after_mb"] > 0:
        print(f"  메모리 (RSS):       {m['mem_after_mb']:.0f} MB ({m['mem_after_mb']/1024:.1f} GB)")
    if "cpu_usage" in m and m["cpu_usage"] > 0:
        print(f"  CPU 사용률:         {m['cpu_usage']:.1f}%")


results = []

def run_benchmark(name, prompt, num_ctx=4096):
    print(f"\n{'='*70}")
    print(f"  벤치마크: {name}")
    print(f"{'='*70}")
    print(f"  프롬프트: {prompt[:80]}...")

    # 논스트리밍 (상세 메트릭)
    m = api_call_detailed(prompt, stream=False, num_ctx=num_ctx)
    print_metrics("결과", m)
    print(f"  응답 미리보기: {m['response'][:150].strip()}...")

    results.append({"name": name, **{k: v for k, v in m.items() if k != "response"}})
    return m


# ── Ollama 서비스 확인 ──

print("\nOllama 서비스 확인 중...")
try:
    urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=5)
    print("  Ollama 서비스 정상 실행 중\n")
except Exception:
    print("  [ERROR] Ollama가 실행 중이 아닙니다. 'brew services start ollama' 로 시작하세요.")
    sys.exit(1)

chip, total_mem = get_system_info()

# ── 웜업 (모델 로딩) ──

print("모델 웜업 중 (첫 로딩은 느릴 수 있음)...")
warmup_start = time.time()
api_call_detailed("Hi", stream=False)
warmup_time = time.time() - warmup_start
print(f"  웜업 완료: {warmup_time:.1f}s\n")

# ── 벤치마크 1: 짧은 응답 (속도 측정) ──

run_benchmark(
    "짧은 응답 (산술)",
    "What is 2+2? Answer with just the number.",
)

# ── 벤치마크 2: 중간 길이 응답 ──

run_benchmark(
    "중간 응답 (설명)",
    "Python의 GIL(Global Interpreter Lock)이 무엇인지 3문장으로 설명해줘.",
)

# ── 벤치마크 3: 코드 생성 ──

run_benchmark(
    "코드 생성 (Python)",
    "Python으로 LRU Cache를 직접 구현해줘. OrderedDict를 사용하지 않고 doubly linked list + dict로 구현해. 코드만 출력해.",
)

# ── 벤치마크 4: 긴 응답 생성 ──

run_benchmark(
    "긴 응답 (상세 설명)",
    "마이크로서비스 아키텍처의 장단점, 모놀리식 대비 트레이드오프, 실제 적용 시 주의사항을 상세히 설명해줘. 최소 500자 이상.",
)

# ── 벤치마크 5: 대형 컨텍스트 ──

run_benchmark(
    "대형 컨텍스트 (16K)",
    "다음 코드의 버그를 찾아줘:\n" + """
class ThreadPool:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.tasks = []
        self.workers = []
        self.lock = threading.Lock()
        self.shutdown_flag = False

    def submit(self, fn, *args, **kwargs):
        with self.lock:
            if self.shutdown_flag:
                raise RuntimeError("Pool is shut down")
            future = Future()
            self.tasks.append((fn, args, kwargs, future))
            if len(self.workers) < self.max_workers:
                w = threading.Thread(target=self._worker)
                w.daemon = True
                w.start()
                self.workers.append(w)
            return future

    def _worker(self):
        while True:
            with self.lock:
                if not self.tasks:
                    if self.shutdown_flag:
                        return
                    continue  # busy wait bug
                task = self.tasks.pop(0)
            fn, args, kwargs, future = task
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    def shutdown(self, wait=True):
        self.shutdown_flag = True
        if wait:
            for w in self.workers:
                w.join()
""",
    num_ctx=16384,
)

# ── 벤치마크 6: 스트리밍 TTFT 측정 ──

print(f"\n{'='*70}")
print(f"  벤치마크: 스트리밍 TTFT 측정")
print(f"{'='*70}")

stream_m = api_call_detailed("Hello, introduce yourself in one sentence.", stream=True)
print(f"  TTFT: {stream_m['ttft']:.2f}s")
print(f"  총 시간: {stream_m['wall_time']:.2f}s")
print(f"  출력 토큰: {stream_m['eval_tokens']}")
print(f"  생성 속도: {stream_m['tokens_per_sec']:.1f} tokens/s")
results.append({"name": "스트리밍 TTFT", **{k: v for k, v in stream_m.items() if k != "response"}})

# ── 최종 리소스 사용량 ──

print(f"\n{'='*70}")
print(f"  리소스 사용량 (추론 중)")
print(f"{'='*70}")

mem = get_memory_usage()
cpu = get_cpu_usage()
print(f"  Ollama 프로세스 RSS: {mem:.0f} MB ({mem/1024:.1f} GB)")
print(f"  Ollama CPU 사용률:   {cpu:.1f}%")
print(f"  시스템 전체 메모리:   {total_mem:.0f} GB")
if mem > 0:
    print(f"  모델 메모리 점유율:  {mem/1024/total_mem*100:.1f}%")

# ── 비용 분석 ──

print(f"\n{'='*70}")
print(f"  비용 분석")
print(f"{'='*70}")

# 평균 토큰/초 계산
avg_tps = sum(r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0) / len([r for r in results if r["tokens_per_sec"] > 0])
avg_prompt_tps = sum(r.get("prompt_tokens_per_sec", 0) for r in results if r.get("prompt_tokens_per_sec", 0) > 0)
prompt_count = len([r for r in results if r.get("prompt_tokens_per_sec", 0) > 0])
if prompt_count > 0:
    avg_prompt_tps /= prompt_count

# 전력 소비 추정 (M4 Max TDP ~80W, 추론 시 약 40-60W 추정)
power_watts = 50  # M4 Max 추론 시 평균 전력 추정
electricity_cost_kwh = 120  # 한국 평균 전기요금 (원/kWh), 약 $0.09

# MacBook Pro M4 Max 가격 기준
macbook_price_krw = 5_990_000  # 128GB M4 Max 기본 가격 (원)
depreciation_years = 4
hours_per_day = 8
days_per_year = 365

# 연간 비용
annual_depreciation = macbook_price_krw / depreciation_years
annual_electricity = power_watts / 1000 * hours_per_day * days_per_year * electricity_cost_kwh
annual_total = annual_depreciation + annual_electricity

# 시간당 비용
hourly_cost = annual_total / (hours_per_day * days_per_year)

# 토큰당 비용
tokens_per_hour = avg_tps * 3600
cost_per_token = hourly_cost / tokens_per_hour
cost_per_1k_tokens = cost_per_token * 1000
cost_per_1m_tokens = cost_per_token * 1_000_000

print(f"\n  [평균 성능]")
print(f"  평균 생성 속도:     {avg_tps:.1f} tokens/s")
if avg_prompt_tps > 0:
    print(f"  평균 입력 처리:     {avg_prompt_tps:.1f} tokens/s")
print(f"  시간당 토큰:        {tokens_per_hour:,.0f} tokens/hour")

print(f"\n  [비용 추정 (로컬)]")
print(f"  하드웨어 감가상각:  {annual_depreciation:,.0f} 원/년 (4년 기준)")
print(f"  전기요금 (추정):    {annual_electricity:,.0f} 원/년 ({power_watts}W, {hours_per_day}h/day)")
print(f"  총 연간 비용:       {annual_total:,.0f} 원/년")
print(f"  시간당 비용:        {hourly_cost:.1f} 원/시간")
print(f"  1K 토큰당 비용:     {cost_per_1k_tokens:.4f} 원 (${cost_per_1k_tokens/1350:.6f})")
print(f"  1M 토큰당 비용:     {cost_per_1m_tokens:.1f} 원 (${cost_per_1m_tokens/1350:.4f})")

print(f"\n  [클라우드 API 비교 (출력 토큰 기준)]")
cloud_prices = {
    "GPT-4o":           {"input": 2.50, "output": 10.00},
    "GPT-4o-mini":      {"input": 0.15, "output": 0.60},
    "Claude Sonnet 4":  {"input": 3.00, "output": 15.00},
    "Claude Haiku 3.5": {"input": 0.80, "output": 4.00},
    "Gemini 1.5 Pro":   {"input": 1.25, "output": 5.00},
    "Llama 3.1 70B (Groq)": {"input": 0.59, "output": 0.79},
}

local_cost_usd_1m = cost_per_1m_tokens / 1350

header = f"  {'모델':<25} {'입력 $/1M':>10} {'출력 $/1M':>10} {'로컬 대비':>10}"
print(header)
print(f"  {'-'*55}")
local_cost_str = f"${local_cost_usd_1m:.4f}"
print(f"  {'로컬 (devstral-2:123b)':<25} {'N/A':>10} {local_cost_str:>10} {'기준':>10}")
for name, prices in cloud_prices.items():
    ratio = prices["output"] / local_cost_usd_1m if local_cost_usd_1m > 0 else 0
    in_str = f"${prices['input']:.2f}"
    out_str = f"${prices['output']:.2f}"
    ratio_str = f"{ratio:.0f}x"
    print(f"  {name:<25} {in_str:>10} {out_str:>10} {ratio_str:>10}")

# ── 종합 결과표 ──

print(f"\n{'='*70}")
print(f"  종합 벤치마크 결과")
print(f"{'='*70}")
print(f"  {'테스트':<30} {'시간':>8} {'토큰':>6} {'속도':>12} {'TTFT':>8}")
print(f"  {'-'*64}")
for r in results:
    ttft_str = f"{r['ttft']:.2f}s" if "ttft" in r and r["ttft"] > 0 else "-"
    print(f"  {r['name']:<30} {r['wall_time']:>7.2f}s {r['eval_tokens']:>5} {r['tokens_per_sec']:>8.1f} t/s {ttft_str:>8}")

print(f"\n  평균 생성 속도: {avg_tps:.1f} tokens/s")
print(f"  모델: {MODEL} | 칩: {chip} | RAM: {total_mem:.0f}GB")
print(f"{'='*70}\n")

# JSON으로 결과 저장
output = {
    "system": {"chip": chip, "memory_gb": total_mem, "model": MODEL},
    "benchmarks": results,
    "avg_tokens_per_sec": avg_tps,
    "cost_per_1m_tokens_krw": cost_per_1m_tokens,
    "cost_per_1m_tokens_usd": local_cost_usd_1m,
}
with open("benchmark_results.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("결과가 benchmark_results.json 에 저장되었습니다.")
