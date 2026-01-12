# Validate Benchmark UI After FFI Changes

**Goal:** Ensure the benchmark UI still works after FFI implementation.

---

## 🎯 Your Task

Validate that the benchmark UI is functional and not broken by recent FFI changes.

## ✅ Validation Steps

### 1. Start the Backend

```bash
cd ui/backend
python app.py
# Should start FastAPI on http://localhost:8000
```

**Expected:** Server starts without errors

### 2. Test API Health

```bash
curl http://localhost:8000/api/health
```

**Expected:** `{"status":"healthy"}`

### 3. Test Mojo Benchmark Endpoint

```bash
curl -X POST http://localhost:8000/api/benchmark/mojo \
  -H "Content-Type: application/json" \
  -d '{"duration": 1, "iterations": 3}'
```

**Expected:** JSON response with:
- `implementation: "mojo-audio"`
- `avg_time_ms: ~1.3`
- `success: true`

### 4. Test Librosa Benchmark Endpoint

```bash
curl -X POST http://localhost:8000/api/benchmark/librosa \
  -H "Content-Type: application/json" \
  -d '{"duration": 1, "iterations": 3}'
```

**Expected:** JSON response with:
- `implementation: "librosa"`
- `avg_time_ms: ~15`
- `success: true`

### 5. Test Comparison Endpoint

```bash
curl -X POST http://localhost:8000/api/benchmark/both \
  -H "Content-Type: application/json" \
  -d '{"duration": 1, "iterations": 3}'
```

**Expected:** JSON with:
- `mojo: {...}`
- `librosa: {...}`
- `speedup_factor: >1.0`
- `mojo_is_faster: true`

### 6. Test Frontend

Open browser to: `http://localhost:8000`

**Check:**
- [ ] Page loads without errors
- [ ] UI renders correctly (Raycast-inspired design)
- [ ] Configuration options visible
- [ ] Click "Run Benchmark" button
- [ ] Results display after ~30 seconds
- [ ] Chart shows optimization journey
- [ ] Speedup calculation correct

---

## 🐛 If Anything Breaks

**Backend errors:**
- Check `ui/backend/requirements.txt` dependencies installed
- Verify Python 3.10+
- Check pixi environment active

**Benchmark failures:**
- Test manually: `pixi run bench-optimized`
- Check paths in `ui/backend/run_benchmark.py`
- Verify Mojo and Python accessible

**Frontend issues:**
- Check browser console for errors
- Verify static files served correctly
- Check API endpoints responding

---

## 📋 Report

After validation, report:
- [ ] Backend starts successfully
- [ ] All 5 API endpoints work
- [ ] Frontend loads and renders
- [ ] Benchmark execution works end-to-end
- [ ] Results are accurate

If all pass: ✅ UI is intact
If any fail: 🐛 Note the issue for fixing
