# Demo Mode Setup Guide

## Overview

Demo mode creates a fully interactive benchmark UI without needing a server. It uses pre-computed benchmark data with smart interpolation and realistic variance.

## Features

✅ **24 Pre-computed Configs** - All duration/FFT/BLAS combinations
✅ **Smart Interpolation** - Run count (1-100) interpolated from base (20 runs)
✅ **Realistic Variance** - ±3-5% variance simulates real benchmark behavior
✅ **Dynamic Loading** - Loading time scales with config complexity (1.5-8s)
✅ **Fully Interactive** - Users can change all parameters

---

## Setup Steps

### 1. Generate Benchmark Data

Run all 24 benchmark configurations:

```bash
# From repo root
python ui/backend/generate_demo_data.py

# Expected output:
# 🚀 Generating 24 benchmark configurations...
# [1/24] 1s_256fft_mkl
#   Running mojo benchmark: 1s, FFT=256, BLAS=mkl...
#   ✅ 0.85ms ± 0.08ms (1176x realtime)
#   ...
# ✅ Saved 24 benchmark results to: ui/static/data/benchmark_results.json
```

**Time:** ~10-15 minutes (runs all configs with 20 iterations each)

**Output:** `ui/static/data/benchmark_results.json` (~15-20 KB)

### 2. Update Frontend to Use Demo Mode

Edit `ui/frontend/index.html` to include demo mode script:

```html
<!-- Add before closing </body> tag -->
<script src="/static/js/demo-mode.js"></script>
<script>
  // Enable demo mode
  const DEMO_MODE = true;

  // Initialize on page load
  if (DEMO_MODE) {
    window.addEventListener('DOMContentLoaded', async () => {
      const ready = await window.DemoMode.init();
      if (ready) {
        console.log('✅ Demo mode ready');
        showDemoBanner(); // Optional: Show "Demo Mode" indicator
      }
    });
  }
</script>
```

### 3. Update API Calls

Replace real API calls with demo mode:

```javascript
// In main.js, replace runBenchmark() function:

async function runBenchmark() {
  const config = getConfigFromUI(); // Your existing function

  if (DEMO_MODE) {
    // Use demo mode
    showLoadingState();

    const results = await window.DemoMode.runBenchmark(config, (progress) => {
      updateProgressBar(progress.progress);
      updateStatusMessage(progress.message);
    });

    displayResults(results);
    hideLoadingState();
  } else {
    // Use real API
    const response = await fetch('/api/benchmark/both', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    const results = await response.json();
    displayResults(results);
  }
}
```

### 4. Deploy Static Version

Deploy to Vercel/Netlify:

```bash
# Option 1: Vercel
cd ui
vercel

# Option 2: Netlify
netlify deploy --dir=. --prod

# Files to deploy:
# ui/
# ├── frontend/index.html
# ├── static/
# │   ├── css/style.css
# │   ├── js/main.js
# │   ├── js/demo-mode.js
# │   └── data/benchmark_results.json
```

---

## How It Works

### Pre-computed Data (24 configs)

```
Duration × FFT Size × BLAS = 24 configs
────────────────────────────────────
1s  × 256  × mkl       ✅
1s  × 256  × openblas  ✅
1s  × 400  × mkl       ✅
1s  × 400  × openblas  ✅
... (20 more)
```

### Smart Interpolation (Run Count)

User selects 50 runs, but we only have data for 20 runs:

```javascript
// Base data (20 runs): avg=10.5ms, std=1.2ms
// Target: 50 runs

// More runs = more stable (lower std deviation)
stdScale = sqrt(20/50) = 0.63
newStd = 1.2ms × 0.63 = 0.76ms

// Add small variance to avg (simulates different runs)
newAvg = 10.5ms × (1 ± 0.03) = 10.3-10.8ms

// Result: avg=10.65ms, std=0.76ms (interpolated + variance)
```

### Realistic Variance

Each time the benchmark runs, add ±3-5% variance:

```javascript
// Same config, different "runs":
Run 1: 10.52ms ± 1.18ms
Run 2: 10.71ms ± 1.23ms  // Slightly different!
Run 3: 10.48ms ± 1.15ms

// Simulates real benchmark variability
```

### Dynamic Loading Time

Loading time scales with complexity:

```javascript
// Simple: 1s, FFT=256, 1 run
loadingTime = 1.8 seconds

// Medium: 10s, FFT=400, 20 runs
loadingTime = 3.2 seconds

// Complex: 30s, FFT=1024, 100 runs
loadingTime = 6.5 seconds

// Feels realistic without being annoying!
```

---

## Testing Demo Mode Locally

```bash
# 1. Generate data
python ui/backend/generate_demo_data.py

# 2. Serve static files
cd ui
python -m http.server 8000

# 3. Open browser
open http://localhost:8000/frontend/index.html

# 4. Test:
# - Change duration → see loading time increase
# - Change FFT size → see results change
# - Change runs → see std deviation change
# - Run same config twice → slightly different results (variance!)
```

---

## Customization

### Adjust Variance

Edit `demo-mode.js`:

```javascript
// More variance (±8% instead of ±5%)
function addRealisticVariance(value) {
  const variance = (Math.random() - 0.5) * 2 * 0.08;
  return value * (1 + variance);
}
```

### Adjust Loading Time

```javascript
// Faster loading (1-4 seconds instead of 1.5-8)
const totalTime = baseTime * durationFactor * fftFactor * iterationsFactor * blasFactor;
return Math.max(1000, Math.min(4000, withVariance));
```

### Show "Demo Mode" Banner

```javascript
function showDemoBanner() {
  const banner = document.createElement('div');
  banner.className = 'demo-banner';
  banner.innerHTML = `
    🎮 Demo Mode - Using pre-computed results
    <a href="https://github.com/itsdevcoffee/mojo-audio">Run locally</a> for live benchmarks
  `;
  document.body.prepend(banner);
}
```

---

## Production Checklist

- [ ] Generated all 24 benchmark configs
- [ ] Verified `benchmark_results.json` exists and is valid
- [ ] Updated frontend to load demo-mode.js
- [ ] Set `DEMO_MODE = true`
- [ ] Tested all config combinations work
- [ ] Verified variance creates different results each run
- [ ] Checked loading times feel realistic
- [ ] Added optional "Demo Mode" indicator
- [ ] Deployed to Vercel/Netlify
- [ ] Confirmed static site works (no API calls)

---

## FAQ

**Q: Why not just use one pre-computed result?**
A: Users want to experiment! This lets them change params and see realistic results.

**Q: Can people tell it's not real?**
A: The variance and loading simulation make it feel very real. Add a small "Demo Mode" badge if you want transparency.

**Q: What if I update the code and benchmarks change?**
A: Re-run `generate_demo_data.py` to update with new results.

**Q: Can I use this approach for other projects?**
A: Absolutely! This pattern works for any computational demo that's too expensive to run on serverless.

---

**Ready to deploy!** 🚀

Run the generator, integrate demo-mode.js, and you'll have a fully interactive demo that costs $0 to host.
