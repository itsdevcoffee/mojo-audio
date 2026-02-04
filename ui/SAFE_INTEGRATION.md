# Safe Demo Mode Integration

## ✅ Validation Complete

```
✅ benchmark_results.json created successfully
✅ 24 configurations present
✅ File size: 10.7 KB
✅ Speedup range: 1.39x - 3.52x (realistic)
✅ All data valid and ready
```

---

## 🎯 Integration Plan

### Strategy: **Backward-Compatible, Zero Breaking Changes**

The integration:
- ✅ **Preserves existing server mode** - Still works when backend running
- ✅ **No changes to main.js initially** - We wrap it, not replace it
- ✅ **Auto-detects environment** - Works correctly everywhere
- ✅ **Graceful fallback** - Server fails → demo mode
- ✅ **Easy rollback** - Remove 3 script tags = back to original

---

## 📝 Step-by-Step Integration

### Step 1: Add Scripts to `index.html` (Non-Breaking)

**Location:** Right before the closing `</body>` tag, **BEFORE** `main.js`

```html
<!-- Current (line 192): -->
<script src="../static/js/main.js"></script>
</body>

<!-- Change to: -->
<script src="../static/js/config.js"></script>
<script src="../static/js/demo-mode.js"></script>
<script src="../static/js/benchmark-api.js"></script>
<script src="../static/js/main.js"></script>

<!-- Initialize on load -->
<script>
// Initialize benchmark system
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const mode = await window.BenchmarkAPI.init();
        console.log('✅ Benchmark UI initialized in', mode, 'mode');

        // Show mode indicator
        if (mode === 'demo') {
            showDemoModeBanner();
        }

        updateStatusBar(mode);
    } catch (error) {
        console.error('❌ Initialization failed:', error);
        alert('Failed to initialize benchmark UI. Please refresh the page.');
    }
});

// Show demo mode banner
function showDemoModeBanner() {
    const banner = document.createElement('div');
    banner.className = 'demo-banner';
    banner.innerHTML = `
        <span class="demo-badge">🎮 Demo Mode</span>
        <span class="demo-text">
            Pre-computed results with smart interpolation.
            <a href="https://github.com/itsdevcoffee/mojo-audio#building-from-source" target="_blank">
                Run locally
            </a> for live benchmarks.
        </span>
    `;
    document.querySelector('.container').insertBefore(
        banner,
        document.querySelector('.header').nextSibling
    );
}

// Update status bar with mode
function updateStatusBar(mode) {
    const statusBar = document.querySelector('.status-bar');
    const modeText = mode === 'demo' ? 'DEMO_MODE' : 'LIVE_MODE';
    const firstSpan = statusBar.querySelector('span');
    firstSpan.textContent = `• ${modeText}: ${mode.toUpperCase()}`;
}
</script>
</body>
```

### Step 2: Update `main.js` - Replace API Call Only

**Find the `runBenchmark()` function (lines 89-130)**

**Replace the fetch call (lines 104-114) with:**

```javascript
// OLD CODE (remove lines 104-114):
// const response = await fetch(`${API_BASE}/benchmark/both`, {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify(config)
// });
//
// if (!response.ok) {
//     throw new Error('Benchmark failed');
// }
//
// const results = await response.json();

// NEW CODE (add):
const results = await window.BenchmarkAPI.run(config, {
    onProgress: (progress) => {
        // Update loading overlay text with progress
        const loadingText = document.querySelector('.loading-text');
        const loadingSubtext = document.querySelector('.loading-subtext');

        if (progress.message) {
            loadingText.textContent = progress.message;
        }

        if (progress.progress !== undefined) {
            const percent = Math.round(progress.progress * 100);
            loadingSubtext.textContent = `${percent}% complete`;
        }
    }
});
```

**That's it!** Everything else stays the same.

### Step 3: Add Demo Banner CSS

Add to `ui/static/css/style.css`:

```css
/* Demo Mode Banner */
.demo-banner {
    background: linear-gradient(135deg,
        rgba(102, 126, 234, 0.08),
        rgba(118, 75, 162, 0.08)
    );
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    padding: 14px 20px;
    margin: 24px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    backdrop-filter: blur(10px);
}

.demo-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 5px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

.demo-text {
    color: #e0e0e0;
    line-height: 1.5;
}

.demo-text a {
    color: #00ff9d;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.2s;
}

.demo-text a:hover {
    text-decoration: underline;
    color: #00ffb3;
}
```

---

## 🔀 Mode Toggle Behavior

### Automatic Detection

| Environment | Detection | Mode | Backend Required? |
|-------------|-----------|------|-------------------|
| `vercel.app` | Hostname check | **Demo** | No |
| `netlify.app` | Hostname check | **Demo** | No |
| `github.io` | Hostname check | **Demo** | No |
| `localhost:8000` | Port + health check | **Server** | Yes |
| `localhost:3000` | Port (static) | **Demo** | No |
| File opened | `file://` | **Demo** | No |

### Manual Override (Optional)

Edit `config.js` line 27:

```javascript
// Force demo mode everywhere
mode: 'demo',

// Force server mode everywhere
mode: 'server',

// Auto-detect (recommended)
mode: 'auto',
```

---

## 🧪 Testing Protocol

### Test 1: Demo Mode (Static)

```bash
# Serve static files
cd ui
python -m http.server 3000

# Open: http://localhost:3000/frontend/index.html
```

**Expected:**
- ✅ Demo banner appears
- ✅ Status bar shows "DEMO_MODE: DEMO"
- ✅ Click "Run Benchmark"
- ✅ Loading shows 2-6s (varies by config)
- ✅ Results appear with realistic values
- ✅ Run again → slightly different results (variance)
- ✅ Change runs to 50 → different std deviation
- ✅ Change duration → different loading time

### Test 2: Server Mode (Live API)

```bash
# Terminal 1: Start backend
cd ui/backend
python app.py

# Terminal 2: Open browser
# Open: http://localhost:8000
```

**Expected:**
- ✅ No demo banner
- ✅ Status bar shows "LIVE_MODE: SERVER"
- ✅ Console shows "✅ Server mode initialized"
- ✅ Click "Run Benchmark"
- ✅ Real API calls happen (check Network tab)
- ✅ Results are actual benchmark data

### Test 3: Fallback Behavior

```bash
# Start server WITHOUT backend running
cd ui
python -m http.server 8000

# Open: http://localhost:8000/frontend/index.html
```

**Expected:**
- ✅ Console shows "Backend not available"
- ✅ Console shows "Falling back to demo mode"
- ✅ Demo banner appears
- ✅ Benchmarks still work (using demo data)

---

## 🛡️ Safety Features

### 1. Non-Breaking Changes
- Existing code still works
- Only wraps the API call
- Doesn't modify displayResults()
- Doesn't modify UI structure

### 2. Graceful Degradation
```javascript
// Priority order:
1. Try server mode (if detected)
2. Fall back to demo mode (if server fails)
3. Show error (if demo data missing)
```

### 3. Clear Logging
```javascript
console.log('🔧 Configuration: Demo mode');
console.log('🎮 Demo mode initialized');
console.log('🎯 Running benchmark in demo mode');
```

### 4. Error Boundaries
- Demo mode fails → show error message
- Server mode fails → try demo mode fallback
- Both fail → clear error to user

---

## 📋 Pre-Integration Checklist

- [x] Benchmark data generated (benchmark_results.json)
- [x] Data validated (24 configs, all valid)
- [ ] Backup current `index.html` and `main.js`
- [ ] Add 3 script tags to `index.html`
- [ ] Update fetch call in `main.js` runBenchmark()
- [ ] Add demo banner CSS
- [ ] Test demo mode (static server)
- [ ] Test server mode (FastAPI backend)
- [ ] Test fallback (no backend)
- [ ] Commit changes
- [ ] Deploy to Vercel

---

## 🔄 Rollback Plan (If Needed)

If anything breaks:

```bash
# 1. Revert index.html changes
git checkout HEAD -- ui/frontend/index.html

# 2. Revert main.js changes
git checkout HEAD -- ui/static/js/main.js

# 3. Remove new files (optional)
rm ui/static/js/{config,demo-mode,benchmark-api}.js

# Done - back to original working state
```

---

## 🚀 Deployment Environments

### Development (localhost)
```bash
# Server mode
cd ui/backend && python app.py
# Open: http://localhost:8000

# Demo mode
cd ui && python -m http.server 3000
# Open: http://localhost:3000/frontend/index.html
```

### Production - Demo (Vercel)
```bash
cd ui
vercel --prod

# Auto-detects demo mode from hostname
# Result: https://mojo-audio.vercel.app
```

### Production - Server (Fly.io)
```bash
fly deploy

# Serves backend + static files
# Result: https://mojo-audio.fly.dev
```

---

## 💡 Pro Tips

1. **During development:** Keep backend running (server mode)
2. **For demos/sharing:** Deploy static version (demo mode)
3. **For production:** Start with demo, add server if demand warrants
4. **Testing:** Use `mode: 'demo'` in config.js to force demo mode locally

---

Ready to integrate! Start with Step 1 (add scripts to index.html).
