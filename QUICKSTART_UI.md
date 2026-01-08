# 🚀 Quickstart: mojo-audio Benchmark UI

**Your Raycast-inspired benchmark interface is READY!**

---

## ▶️ **Start the UI (2 commands!)**

```bash
cd /home/maskkiller/dev-coffee/repos/mojo-audio

# Start the UI server
pixi run ui
```

**Then open:** http://localhost:8000

---

## 🎨 **What You'll See**

**Clean, Raycast-inspired interface:**

1. **Header** - Gradient orange title, clean subtitle
2. **Configuration Card** - Pill buttons for duration/FFT size
3. **Run Benchmark Button** - Orange gradient, smooth hover
4. **Results Cards** (after running)
   - Side-by-side: librosa vs mojo-audio
   - Big numbers (48px, monospace)
   - Animated progress bars
   - Success badge if mojo wins!
5. **Optimization Journey Chart** - Line graph (476ms → 12ms)
6. **Download Results** button

---

## 🧪 **Try It Out**

### Step 1: Start Server
```bash
pixi run ui
# Server runs at http://localhost:8000
```

### Step 2: Open Browser
Navigate to: http://localhost:8000

### Step 3: Run Benchmark
1. Select "30 seconds" (default)
2. Keep FFT at "400 (Whisper)"
3. Runs: 5 (default)
4. Click **"🚀 Run Benchmark"**

### Step 4: Watch the Magic!
- Loading overlay appears
- Benchmarks run (~30-60 seconds total)
- Results animate in smoothly
- Chart shows optimization journey
- Success badge celebrates victory! 🎉

---

## 📊 **Expected Results**

```
librosa:    15.0ms  ████████████████████
mojo-audio: 12.2ms  ████████████████     ⚡ 23% faster!

Speedup Factor: 1.23x
Throughput: 2,457x realtime
```

**mojo-audio should consistently beat librosa!** 🏆

---

## 🎯 **Features to Try**

### Different Durations
- **1s**: Fast benchmark (~5 seconds)
- **10s**: Medium benchmark (~15 seconds)
- **30s**: Full Whisper benchmark (~30 seconds)

### Different FFT Sizes
- **256**: Smaller FFT
- **400**: Whisper standard (best comparison)
- **512**: Power of 2

### Benchmark Runs
- Fewer runs (3): Faster results
- More runs (10): More accurate average

---

## 🐛 **Troubleshooting**

### Server won't start
```bash
# Make sure you're in mojo-audio directory
cd /home/maskkiller/dev-coffee/repos/mojo-audio

# Check pixi environment
pixi install

# Try again
pixi run ui
```

### Benchmark fails
```bash
# Test benchmarks manually first
pixi run bench-optimized  # Should work
pixi run bench-python     # Should work

# If these work, UI should work too
```

### Browser shows blank page
- Make sure server is running (see terminal output)
- Check http://localhost:8000 (not https!)
- Try hard refresh (Ctrl+Shift+R)

---

## 📸 **Perfect for Screenshots!**

The Raycast-style design makes BEAUTIFUL screenshots for:
- Blog posts
- Social media (Twitter/X, Reddit)
- GitHub README
- Presentations
- Documentation

**Clean, professional, and the data speaks for itself!**

---

## 🎊 **Next Steps**

Once you've tested it:

1. **Take screenshots** of the results
2. **Deploy to Vercel** (free, instant)
   ```bash
   cd ui
   vercel
   ```
3. **Share the link!**
4. **Write blog post** with embedded UI

---

**The UI is LIVE and READY!** 🔥

**Try it now:** `pixi run ui` then visit http://localhost:8000

🎨 **Beautiful Raycast design!**
🏆 **Shows mojo-audio beating Python!**
📊 **Interactive and engaging!**
