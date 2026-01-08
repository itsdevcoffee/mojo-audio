# mojo-audio Benchmark UI

**Raycast-inspired interactive benchmark interface**

Beautiful, clean UI for comparing mojo-audio vs librosa performance.

---

## 🎨 **Design**

**Style:** Raycast + Clean Light aesthetic
- Soft colors, generous whitespace
- Smooth animations and transitions
- Data-first presentation
- Professional and credible

**Features:**
- ⚡ Real-time benchmark execution
- 📊 Interactive optimization journey chart
- 🎯 Configurable parameters (duration, FFT size, iterations)
- 📥 Download results as JSON
- 🏆 Clear winner indication

---

## 🚀 **Quick Start**

### Install Dependencies

```bash
# Install Python dependencies
cd ui/backend
pip install -r requirements.txt

# Or use pixi (if in main repo)
pixi install
```

### Run Locally

```bash
# Start FastAPI backend
cd ui/backend
python app.py

# Backend runs at: http://localhost:8000
# Open browser to: http://localhost:8000
```

### Development Mode

```bash
# Run with auto-reload
uvicorn app:app --reload --port 8000

# Open: http://localhost:8000
```

---

## 📖 **API Endpoints**

### POST `/api/benchmark/mojo`
Run mojo-audio benchmark with -O3 optimization.

**Request:**
```json
{
  "duration": 30,
  "n_fft": 400,
  "hop_length": 160,
  "n_mels": 80,
  "iterations": 3
}
```

**Response:**
```json
{
  "implementation": "mojo-audio",
  "avg_time_ms": 12.2,
  "throughput_realtime": 2457,
  "success": true
}
```

### POST `/api/benchmark/librosa`
Run librosa (Python) benchmark.

### POST `/api/benchmark/both`
Run both and return comparison with speedup factor.

**Response:**
```json
{
  "mojo": {...},
  "librosa": {...},
  "speedup_factor": 1.23,
  "faster_percentage": 23.0,
  "mojo_is_faster": true
}
```

---

## 🎯 **UI Components**

### Configuration Panel
- Radio pills for duration (1s, 10s, 30s)
- FFT size selection (256, 400, 512, 1024)
- Iteration count with +/- buttons
- Primary button to run benchmark

### Results Display
- Side-by-side comparison cards
- Large numbers (Raycast-style)
- Animated progress bars
- Throughput stats
- Success badge when mojo wins

### Optimization Chart
- Line chart showing 476ms → 12ms journey
- Interactive tooltips
- Soft color palette
- Shows each optimization step

### Actions
- Download results (JSON)
- Link to GitHub repo

---

## 🌐 **Deployment**

### Option 1: Vercel (Recommended!)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd ui
vercel

# Follow prompts
# Backend will be serverless functions
```

### Option 2: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway init
railway up

# Automatic deployment from git
```

### Option 3: Docker

```bash
# Build image
docker build -t mojo-audio-ui .

# Run
docker run -p 8000:8000 mojo-audio-ui
```

---

## 🎨 **Customization**

### Colors

Edit `ui/static/css/style.css`:
```css
:root {
    --mojo-orange: #FF8A5B;  /* Change primary color */
    --success-green: #52C93F; /* Change success color */
}
```

### Chart

Edit `ui/static/js/main.js`:
```javascript
const optimizations = [
    { name: 'Your step', time: 100 },
    // Add your optimization steps
];
```

---

## 🧪 **Testing Locally**

```bash
# Terminal 1: Start backend
cd ui/backend
python app.py

# Terminal 2: Test API
curl http://localhost:8000/api/health

# Browser: Open UI
open http://localhost:8000
```

**Expected:**
- Clean, light interface loads
- Configuration options visible
- Click "Run Benchmark"
- Results appear with smooth animations
- Chart shows optimization journey

---

## 📊 **Performance**

**Backend:**
- FastAPI (async, fast!)
- Subprocess calls to Mojo/Python
- ~30s for both benchmarks

**Frontend:**
- Vanilla JS (no framework overhead!)
- Chart.js for visualizations
- Smooth CSS animations
- Responsive design

---

## 🔧 **Troubleshooting**

### Backend won't start
```bash
# Check Python version (3.10+)
python --version

# Install dependencies
pip install -r backend/requirements.txt
```

### Benchmarks fail
```bash
# Ensure mojo-audio repo has pixi environment
pixi install

# Test benchmarks manually
pixi run bench-optimized
pixi run bench-python
```

### CORS errors
Backend has CORS enabled for development.
For production, update `allow_origins` in `app.py`.

---

## 🎊 **Features**

**Implemented:**
- ✅ Raycast-inspired design
- ✅ Real-time benchmarking
- ✅ Interactive charts
- ✅ Configurable parameters
- ✅ Smooth animations
- ✅ Download results
- ✅ Responsive layout

**Future ideas:**
- Toggle individual optimizations on/off
- Historical result tracking
- Share link with pre-filled results
- Embed mode for blog posts

---

**Built with:** FastAPI + Vanilla JS + Chart.js + Raycast aesthetics 🎨

**See it in action:** Run `python backend/app.py` and visit http://localhost:8000
