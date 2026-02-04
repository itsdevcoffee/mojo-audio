# Safe Demo Mode Integration Plan

## ✅ Validation Results

```
✅ All 24 configs generated successfully
✅ Data quality validated
✅ Speedup range: 1.39x - 3.52x (realistic!)
✅ File size: 10.7 KB (tiny!)
```

---

## 🎯 Integration Strategy

### Safe Mode Detection (Auto-Toggle)

The system **automatically detects** the environment:

```javascript
// Vercel/Netlify/GitHub Pages → Demo mode
// localhost:8000 with backend  → Server mode
// localhost without backend    → Demo mode (with warning)
```

**Benefits:**
- ✅ No manual configuration needed
- ✅ Same codebase works everywhere
- ✅ Graceful fallback if backend unavailable
- ✅ Developer-friendly (auto-detects local server)

### Script Load Order (Critical!)

```html
<!-- 1. Configuration first (detects mode) -->
<script src="/static/js/config.js"></script>

<!-- 2. Demo mode (always loaded, but only used if needed) -->
<script src="/static/js/demo-mode.js"></script>

<!-- 3. Unified API (decides which mode to use) -->
<script src="/static/js/benchmark-api.js"></script>

<!-- 4. Your existing UI code -->
<script src="/static/js/main.js"></script>
```

---

## 📝 Integration Steps

### Step 1: Update `index.html`

Add scripts before your existing `main.js`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- existing head content -->
</head>
<body>
    <!-- existing UI content -->

    <!-- Load in order: Config → Demo → API → Main -->
    <script src="/static/js/config.js"></script>
    <script src="/static/js/demo-mode.js"></script>
    <script src="/static/js/benchmark-api.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>
```

### Step 2: Update `main.js` - Replace API Calls

**Find your existing benchmark function** (probably named `runBenchmark()` or similar)

**Replace with:**

```javascript
// OLD CODE (remove):
// async function runBenchmark() {
//     const response = await fetch('/api/benchmark/both', {
//         method: 'POST',
//         body: JSON.stringify(config)
//     });
//     const results = await response.json();
//     displayResults(results);
// }

// NEW CODE (add):
async function runBenchmark() {
    // Get config from UI inputs
    const config = {
        duration: parseInt(document.querySelector('[name="duration"]:checked').value),
        n_fft: parseInt(document.querySelector('[name="fft-size"]:checked').value),
        hop_length: 160,  // Fixed for Whisper
        n_mels: 80,       // Fixed for Whisper
        iterations: parseInt(document.querySelector('#runs-input').value),
        blas_backend: document.querySelector('[name="blas"]:checked').value
    };

    // Show loading state
    showLoadingState();

    try {
        // Use unified API (automatically picks demo or server mode)
        const results = await window.BenchmarkAPI.run(config, {
            onProgress: (progress) => {
                updateProgressBar(progress.progress * 100);
                updateStatusMessage(progress.message);
            }
        });

        // Display results (your existing function)
        displayResults(results);

    } catch (error) {
        showError(error.message);
    } finally {
        hideLoadingState();
    }
}
```

### Step 3: Add Mode Indicator (Optional)

Show users which mode they're in:

```javascript
// Add to your initialization code
async function initializeApp() {
    // Initialize benchmark API
    const mode = await window.BenchmarkAPI.init();

    // Show mode indicator
    if (mode === 'demo' && AppConfig.demo.showBanner) {
        showDemoBanner();
    }
}

function showDemoBanner() {
    const banner = document.createElement('div');
    banner.className = 'demo-banner';
    banner.innerHTML = `
        <span class="demo-badge">🎮 Demo Mode</span>
        <span class="demo-text">
            Using pre-computed results.
            <a href="https://github.com/itsdevcoffee/mojo-audio#building-from-source"
               target="_blank">Run locally</a> for live benchmarks.
        </span>
    `;
    document.querySelector('.container').prepend(banner);
}
```

**CSS for banner:**
```css
.demo-banner {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border: 1px solid #667eea40;
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
}

.demo-badge {
    background: #667eea;
    color: white;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}

.demo-text {
    color: #666;
}

.demo-text a {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
}

.demo-text a:hover {
    text-decoration: underline;
}
```

---

## 🔀 Mode Toggle Matrix

| Environment | Auto-Detected Mode | Behavior |
|-------------|-------------------|----------|
| `vercel.app` | Demo | Pre-computed data, no backend |
| `netlify.app` | Demo | Pre-computed data, no backend |
| `localhost:8000` (backend running) | Server | Live API calls |
| `localhost:8000` (no backend) | Demo (fallback) | Pre-computed data + warning |
| File opened directly (`file://`) | Demo | Pre-computed data |

---

## 🧪 Testing Plan

### Test Demo Mode

```bash
# Serve static files
cd ui
python -m http.server 3000

# Open: http://localhost:3000/frontend/index.html
# Should show: Demo mode banner
# Test: Change all params, run benchmark
# Verify: Results change, loading times vary
```

### Test Server Mode

```bash
# Terminal 1: Start backend
cd ui/backend
python app.py

# Terminal 2: Open UI
open http://localhost:8000

# Should show: No demo banner
# Test: Run benchmark
# Verify: Real API calls happen
```

### Test Auto-Fallback

```bash
# Start on port 8000 WITHOUT backend
cd ui
python -m http.server 8000

# Open: http://localhost:8000/frontend/index.html
# Should: Detect no backend, fall back to demo mode
# Verify: Console shows fallback warning
```

---

## 🚀 Deployment

### Demo Mode (Vercel/Netlify)

```bash
# 1. Ensure data file exists
ls -lh ui/static/data/benchmark_results.json

# 2. Deploy
cd ui
vercel --prod

# Result: Fully interactive demo, $0/month
```

### Server Mode (Fly.io/Railway)

```bash
# Deploy with backend
fly deploy

# Result: Live benchmarks with real Mojo execution
```

---

## 🔒 Safety Features

1. **Automatic Detection** - No manual config needed
2. **Graceful Fallback** - Server fails → demo mode
3. **Health Checks** - Verify backend before use
4. **Timeout Protection** - 2min max for server mode
5. **Error Boundaries** - Catch and display errors nicely
6. **Console Logging** - Clear debug info

---

## 🎨 User Experience

### Demo Mode:
- ✨ Instant deployment
- 🎮 Fully interactive
- ⚡ Fast loading (1.5-8s simulated)
- 🎲 Realistic variance
- 💰 $0 hosting

### Server Mode:
- 🔥 Real Mojo execution
- 📊 Actual performance data
- 🎯 Custom hardware benchmarks
- 💻 Requires infrastructure

**Both modes feel identical to users!** 🎭

---

## 📋 Integration Checklist

- [ ] Scripts added to `index.html` in correct order
- [ ] Existing API calls replaced with `BenchmarkAPI.run()`
- [ ] Loading state handlers implemented
- [ ] Error handling added
- [ ] Demo banner CSS added (optional)
- [ ] Tested in demo mode (static server)
- [ ] Tested in server mode (FastAPI backend)
- [ ] Tested auto-fallback (backend unavailable)
- [ ] Deployed to Vercel/Netlify
- [ ] Verified production deployment works

---

**Ready to integrate?** Follow the steps above to add demo mode to your UI safely!
