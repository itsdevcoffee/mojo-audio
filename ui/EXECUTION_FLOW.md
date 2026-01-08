# Backend Routes & Execution Flow

Complete map of what happens when you run a benchmark in the UI.

---

## 🗺️ **API Routes**

### **1. GET /**
```
Route: http://localhost:8000/
Handler: root()
Returns: index.html (the UI page)
```

### **2. GET /static/{path}**
```
Route: http://localhost:8000/static/css/style.css
Handler: StaticFiles middleware
Returns: CSS, JS, images
```

### **3. POST /api/benchmark/mojo**
```
Route: http://localhost:8000/api/benchmark/mojo
Handler: benchmark_mojo(config)
Input: { duration, n_fft, hop_length, n_mels, iterations }
Returns: { implementation, avg_time_ms, throughput_realtime, ... }
```

### **4. POST /api/benchmark/librosa**
```
Route: http://localhost:8000/api/benchmark/librosa
Handler: benchmark_librosa(config)
Input: { duration, n_fft, hop_length, n_mels, iterations }
Returns: { implementation, avg_time_ms, throughput_realtime, ... }
```

### **5. POST /api/benchmark/both** (Primary Route!)
```
Route: http://localhost:8000/api/benchmark/both
Handler: benchmark_both(config)
Input: { duration, n_fft, hop_length, n_mels, iterations }
Returns: {
  mojo: {...},
  librosa: {...},
  speedup_factor: 1.23,
  faster_percentage: 23.0,
  mojo_is_faster: true
}
```

### **6. GET /api/health**
```
Route: http://localhost:8000/api/health
Handler: health()
Returns: { status: "healthy" }
```

---

## 🔄 **Complete Execution Flow**

### **User Clicks "Run Benchmark"**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND (main.js)                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ runBenchmark() called                                           │
│ ├─ Collect config:                                              │
│ │  ├─ duration: selectedDuration (from toggle buttons)          │
│ │  ├─ n_fft: selectedFFT (from toggle buttons)                 │
│ │  └─ iterations: from number input                            │
│ │                                                                │
│ ├─ Show loading overlay                                         │
│ ├─ Disable run button                                           │
│ │                                                                │
│ └─ POST /api/benchmark/both                                     │
│    Body: {                                                       │
│      duration: 30,                                              │
│      n_fft: 400,                                                │
│      hop_length: 160,  (hardcoded in JS)                        │
│      n_mels: 80,       (hardcoded in JS)                        │
│      iterations: 5                                              │
│    }                                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. BACKEND - benchmark_both() (app.py:166-189)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ async def benchmark_both(config):                               │
│   ├─ await benchmark_mojo(config)                               │
│   │   └─ Returns: mojo_result                                   │
│   │                                                              │
│   ├─ await benchmark_librosa(config)                            │
│   │   └─ Returns: librosa_result                                │
│   │                                                              │
│   ├─ Calculate speedup:                                         │
│   │   speedup = librosa_ms / mojo_ms                            │
│   │   faster_pct = ((librosa - mojo) / librosa) * 100          │
│   │                                                              │
│   └─ Return comparison                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           ↓                            ↓
    ┌──────────┐                  ┌──────────┐
    │  MOJO    │                  │ LIBROSA  │
    └──────────┘                  └──────────┘

---

## 🔥 **MOJO BENCHMARK FLOW (Detailed!)**

```
┌─────────────────────────────────────────────────────────────────┐
│ 3. benchmark_mojo() (app.py:62-106)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Receives: config { duration, n_fft, hop_length, n_mels, iters }│
│                                                                 │
│ Builds command:                                                 │
│ cmd = [                                                         │
│   "python",                                                     │
│   "ui/backend/run_benchmark.py",                                │
│   "mojo",                                                       │
│   str(config.duration),      # e.g., "30"                       │
│   str(config.iterations),    # e.g., "5"                        │
│   str(config.n_fft),         # e.g., "400"                      │
│   str(config.hop_length),    # e.g., "160"                      │
│   str(config.n_mels)         # e.g., "80"                       │
│ ]                                                                │
│                                                                 │
│ subprocess.run(cmd, cwd=REPO_ROOT, timeout=120s)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. run_benchmark.py - Python Wrapper                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Parse args:                                                     │
│   sys.argv[1] = "mojo"                                          │
│   sys.argv[2] = "30"      (duration)                            │
│   sys.argv[3] = "5"       (iterations)                          │
│   sys.argv[4] = "400"     (n_fft)                               │
│   sys.argv[5] = "160"     (hop_length)                          │
│   sys.argv[6] = "80"      (n_mels)                              │
│                                                                 │
│ Call: benchmark_mojo_single(30, 5, 400, 160, 80)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. benchmark_mojo_single() (run_benchmark.py:11-66)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ STEP 1: Generate Mojo code dynamically                          │
│ ═══════════════════════════════════════                        │
│                                                                 │
│ mojo_code = f"""                                                │
│ from audio import mel_spectrogram                               │
│ from time import perf_counter_ns                                │
│                                                                 │
│ fn main() raises:                                               │
│     var audio = List[Float32]()                                 │
│     for _ in range({30 * 16000}):  # 480,000 samples           │
│         audio.append(0.1)                                       │
│                                                                 │
│     # Warmup                                                    │
│     _ = mel_spectrogram(audio,                                  │
│         n_fft={400}, hop_length={160}, n_mels={80})            │
│                                                                 │
│     # Benchmark                                                 │
│     var start = perf_counter_ns()                               │
│     for _ in range({5}):                                        │
│         _ = mel_spectrogram(audio,                              │
│             n_fft={400}, hop_length={160}, n_mels={80})        │
│     var end = perf_counter_ns()                                 │
│                                                                 │
│     var avg_ms = (end - start) / {5} / 1_000_000.0             │
│     print(avg_ms)                                               │
│ """                                                             │
│                                                                 │
│ STEP 2: Write to temp file                                      │
│ ════════════════════════                                        │
│ with open('/tmp/mojo_bench_temp.mojo', 'w') as f:              │
│     f.write(mojo_code)                                          │
│                                                                 │
│ STEP 3: Compile & Run with -O3 (CRITICAL!)                      │
│ ════════════════════════════════════════════                   │
│ subprocess.run([                                                │
│     'pixi', 'run', '-e', 'default',                             │
│     'mojo',                                                     │
│     '-O3',           # ← COMPILER OPTIMIZATION FLAG             │
│     '-I', 'src',                                                │
│     '/tmp/mojo_bench_temp.mojo'                                 │
│ ], cwd='/home/maskkiller/dev-coffee/repos/mojo-audio')         │
│                                                                 │
│ What happens:                                                   │
│ ├─ pixi activates conda environment                             │
│ ├─ mojo compiler invoked with -O3                               │
│ ├─ Compiles temp file (aggressive optimizations!)               │
│ ├─ Executes compiled binary                                     │
│ ├─ Benchmark runs inside compiled code:                         │
│ │  ├─ Create audio (480k samples)                               │
│ │  ├─ Warmup: mel_spectrogram() once                           │
│ │  ├─ Start timer                                               │
│ │  ├─ Loop: mel_spectrogram() 5 times                          │
│ │  ├─ End timer                                                 │
│ │  ├─ Calculate average                                         │
│ │  └─ Print result                                              │
│ └─ Output: "12.345\n"                                           │
│                                                                 │
│ STEP 4: Parse output                                            │
│ ════════════════════                                            │
│ avg_time = float(result.stdout.strip())  # "12.345"            │
│                                                                 │
│ STEP 5: Return                                                   │
│ ═══════════                                                     │
│ return 12.345  (in milliseconds)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    Backend returns to
                    benchmark_both()

---

## 🐍 **LIBROSA BENCHMARK FLOW (Detailed!)**

```
┌─────────────────────────────────────────────────────────────────┐
│ 6. benchmark_librosa() (app.py:109-163)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Same subprocess pattern:                                         │
│ cmd = [                                                         │
│   "python",                                                     │
│   "ui/backend/run_benchmark.py",                                │
│   "librosa",                                                    │
│   "30", "5", "400", "160", "80"                                 │
│ ]                                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. benchmark_librosa_single() (run_benchmark.py:68-104)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ STEP 1: Create audio                                            │
│ ════════════════════                                            │
│ audio = np.random.rand(30 * 16000).astype(np.float32) * 0.1    │
│ # 480,000 samples                                               │
│                                                                 │
│ STEP 2: Warmup                                                   │
│ ═══════════                                                     │
│ _ = librosa.feature.melspectrogram(                             │
│     y=audio, sr=16000,                                          │
│     n_fft=400, hop_length=160, n_mels=80                        │
│ )                                                                │
│                                                                 │
│ STEP 3: Benchmark loop                                          │
│ ════════════════════                                            │
│ times = []                                                       │
│ for _ in range(5):                                              │
│     start = time.perf_counter()                                 │
│     _ = librosa.feature.melspectrogram(...)                     │
│     end = time.perf_counter()                                   │
│     times.append((end - start) * 1000)  # ms                    │
│                                                                 │
│ STEP 4: Calculate average                                       │
│ ═════════════════════                                           │
│ avg = np.mean(times)  # Average of 5 runs                       │
│                                                                 │
│ STEP 5: Return                                                   │
│ ═══════════                                                     │
│ return 14.567  (in milliseconds)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

---

## ⚡ **CRITICAL: Where Compilation Happens**

### **MOJO Path:**
```
User clicks → Backend → run_benchmark.py → Generate code → Write /tmp/*.mojo

                            ↓

    pixi run mojo -O3 /tmp/mojo_bench_temp.mojo
           ↓
    COMPILATION PHASE (NOT TIMED):
    ├─ Parse Mojo code
    ├─ Apply -O3 optimizations:
    │  ├─ Loop unrolling
    │  ├─ Inline expansion
    │  ├─ Vectorization
    │  ├─ Dead code elimination
    │  └─ Constant folding
    └─ Generate optimized binary
           ↓
    EXECUTION PHASE (TIMED):
    ├─ Create audio
    ├─ Warmup: mel_spectrogram() ← NOT TIMED
    ├─ Start perf_counter_ns()    ← TIMING STARTS
    ├─ Loop 5x: mel_spectrogram()
    ├─ End perf_counter_ns()      ← TIMING ENDS
    └─ Print average
```

**KEY POINT:** Compilation time is **excluded** from benchmark results!

### **LIBROSA Path:**
```
User clicks → Backend → run_benchmark.py

                            ↓

    Pure Python execution (no compilation)
    ├─ Import librosa (already loaded)
    ├─ Create audio
    ├─ Warmup: librosa.feature.melspectrogram() ← NOT TIMED
    ├─ Loop 5x:
    │  ├─ Start time.perf_counter()  ← TIMING PER ITERATION
    │  ├─ librosa.feature.melspectrogram()
    │  ├─ End time.perf_counter()
    │  └─ Record time
    └─ Average all times
```

**KEY POINT:** No compilation, but **each iteration is timed separately**!

---

## 🎯 **Sources of Performance Variance**

### **Why Results Vary Run-to-Run:**

#### 1. **Mojo Compilation Variance** ❌ (Excluded from timing!)
- Compilation happens before timing starts
- Not a factor in variance

#### 2. **CPU Thermal State** ✅ (MAJOR FACTOR!)
```
Cold CPU:  Higher boost clocks → Faster
Hot CPU:   Thermal throttling → Slower

Variance: ±10-20% possible!
```

#### 3. **System Load** ✅ (MODERATE FACTOR!)
```
Background processes competing for CPU
Cache pressure from other apps
Variance: ±5-15%
```

#### 4. **Parallelization Scheduling** ✅ (MOJO SPECIFIC!)
```
Mojo uses: parallelize[] across all cores (16 cores)

Scheduler variance:
- Thread creation overhead
- Load balancing decisions
- Core migration

Variance: ±5-10%
```

#### 5. **NumPy/librosa Caching** ✅ (PYTHON SPECIFIC!)
```
NumPy may cache:
- FFT plans (FFTW wisdom)
- Memory allocations
- JIT-compiled functions

Variance: ±5-10%
```

#### 6. **Low Iteration Count** ✅ (MAJOR FACTOR!)
```
Current: 3-5 iterations
Problem: Small sample size = high variance

Solution: 10-20 iterations for stable average
```

---

## 🔬 **Measurement Methodology Comparison**

### **MOJO:**
```mojo
# INSIDE compiled binary (pure Mojo timing)
var start = perf_counter_ns()
for _ in range(5):
    _ = mel_spectrogram(audio, ...)  # All 5 runs timed together
var end = perf_counter_ns()

avg = (end - start) / 5
```

**Characteristics:**
- All iterations in one timing block
- Compiled code (optimized)
- Cache warm after first iteration
- Lower measurement overhead

### **LIBROSA:**
```python
# Python wrapper (per-iteration timing)
times = []
for _ in range(5):
    start = time.perf_counter()  # Time each separately
    _ = librosa.feature.melspectrogram(...)
    end = time.perf_counter()
    times.append(end - start)

avg = np.mean(times)
```

**Characteristics:**
- Each iteration timed separately
- Python interpreter overhead (minimal)
- Can detect per-run variance
- Slightly higher measurement overhead

---

## ⚠️ **Why Performance is Close/Variable**

### **Expected with -O3:**
```
30s audio, 400 FFT:
- mojo: 10-12ms (consistently)
- librosa: 15ms (consistently)
- Gap: 25-40% faster
```

### **What You're Seeing:**
```
Run 1: librosa 2ms faster (???)
Run 2: mojo 2.5ms faster
```

**This suggests one of:**

1. **-O3 not actually being used** (would explain similar performance)
2. **Extreme system variance** (unlikely to flip winner!)
3. **Small sample size** (3-5 runs = high variance)
4. **Different FFT sizes tested** (512 is slower for mojo)

---

## 🔧 **Diagnostic Steps**

### **1. Verify -O3 is Working:**
```bash
cd /home/maskkiller/dev-coffee/repos/mojo-audio

# Test wrapper directly
python ui/backend/run_benchmark.py mojo 30 10 400 160 80

# Should consistently show ~10-12ms
```

### **2. Check Compilation Output:**
Look at `/tmp/mojo_bench_temp.mojo` - verify the generated code looks correct.

### **3. Increase Iterations:**
Try 10-20 iterations instead of 5 for more stable results.

### **4. Test Command-Line Baseline:**
```bash
# Our proven benchmark
pixi run bench-optimized

# Should show ~10-12ms for 30s
# If UI shows different, something is wrong!
```

---

## 💡 **Most Likely Issue**

**My guess:** The **pixi run mojo -O3** command in the wrapper might not be working correctly!

Let me check if pixi is properly invoking mojo with -O3...

Actually, the command is:
```python
['pixi', 'run', '-e', 'default', 'mojo', '-O3', '-I', 'src', ...]
```

This should work, but **pixi run** might not pass flags correctly!

**Better approach:** Use direct mojo path from pixi environment!

---

**Want me to investigate and fix the -O3 issue?** That's likely why performance is inconsistent! 🔍