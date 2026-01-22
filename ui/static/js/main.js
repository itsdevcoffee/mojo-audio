// mojo-audio Benchmark UI - Dev Coffee Brand Theme
// Dark terminal aesthetic with neon mint green accents

const API_BASE = 'http://localhost:8000/api';
const STORAGE_KEY = 'mojo_audio_benchmark_history';
const HOP_LENGTH = 160;  // Whisper default
const SAMPLE_RATE = 16000;

// UI State
let selectedDuration = 30;
let selectedFFT = 400;
let selectedSignal = 'chirp';
let benchmarkResults = null;
let optimizationChart = null;

// Signal type definitions (will be populated from API)
let signalTypes = [
    { id: 'chirp', name: 'Chirp', description: 'Linear sweep 20Hz-8000Hz', icon: '📈', recommended: true },
    { id: 'random', name: 'Random', description: 'Seeded random noise', icon: '🎲', has_seed: true },
    { id: 'sine', name: 'Sine 440Hz', description: 'Pure A4 tone', icon: '〰️' },
    { id: 'white_noise', name: 'White Noise', description: 'Random each run', icon: '📻' },
    { id: 'multi_tone', name: 'Multi-Tone', description: 'Speech formants', icon: '🗣️' }
];

// URL Parameter Sync
function updateURLParams() {
    const params = new URLSearchParams();

    // Only add non-default values to keep URL clean
    if (selectedDuration !== 30) params.set('duration', selectedDuration);
    if (selectedFFT !== 400) params.set('fft', selectedFFT);

    const runs = parseInt(document.getElementById('runs').value);
    if (runs !== 20) params.set('runs', runs);

    if (selectedSignal !== 'chirp') params.set('signal', selectedSignal);

    const seed = parseInt(document.getElementById('seedInput').value);
    if (seed !== 42) params.set('seed', seed);

    const warmup = parseInt(document.getElementById('warmupRuns').value);
    if (warmup !== 5) params.set('warmup', warmup);

    const stableMode = document.getElementById('stableMode').checked;
    if (stableMode) {
        params.set('stable', '1');
        const stableRuns = parseInt(document.getElementById('stableRuns').value);
        if (stableRuns !== 5) params.set('stableRuns', stableRuns);
    }

    // Update URL without reload
    const newURL = params.toString()
        ? `${window.location.pathname}?${params.toString()}`
        : window.location.pathname;
    window.history.replaceState({}, '', newURL);
}

function loadFromURLParams() {
    const params = new URLSearchParams(window.location.search);

    // Duration
    if (params.has('duration')) {
        const duration = parseInt(params.get('duration'));
        if ([1, 10, 30].includes(duration)) {
            selectedDuration = duration;
            // Update buttons
            document.querySelectorAll('.toggle-btn[onclick*="selectDuration"]').forEach(btn => {
                const isMatch = btn.textContent.trim() === `${duration}s`;
                btn.classList.toggle('active', isMatch);
            });
        }
    }

    // FFT Size
    if (params.has('fft')) {
        const fft = parseInt(params.get('fft'));
        if ([256, 400, 512, 1024].includes(fft)) {
            selectedFFT = fft;
            // Update buttons
            document.querySelectorAll('.toggle-btn[onclick*="selectFFT"]').forEach(btn => {
                const isMatch = btn.textContent.trim() === `${fft}`;
                btn.classList.toggle('active', isMatch);
            });
        }
    }

    // Runs
    if (params.has('runs')) {
        const runs = parseInt(params.get('runs'));
        if (runs >= 1 && runs <= 30) {
            document.getElementById('runs').value = runs;
        }
    }

    // Signal type
    if (params.has('signal')) {
        const signal = params.get('signal');
        if (signalTypes.some(s => s.id === signal)) {
            selectedSignal = signal;
        }
    }

    // Seed
    if (params.has('seed')) {
        const seed = parseInt(params.get('seed'));
        if (seed >= 0 && seed <= 999999) {
            document.getElementById('seedInput').value = seed;
        }
    }

    // Warmup
    if (params.has('warmup')) {
        const warmup = parseInt(params.get('warmup'));
        if (warmup >= 0 && warmup <= 20) {
            document.getElementById('warmupRuns').value = warmup;
        }
    }

    // Stable mode
    if (params.has('stable') && params.get('stable') === '1') {
        document.getElementById('stableMode').checked = true;
        document.getElementById('stableRunsContainer').style.display = 'flex';

        if (params.has('stableRuns')) {
            const stableRuns = parseInt(params.get('stableRuns'));
            if (stableRuns >= 2 && stableRuns <= 10) {
                document.getElementById('stableRuns').value = stableRuns;
            }
        }
    }
}

// Toggle Functions
function selectDuration(duration, btn) {
    selectedDuration = duration;
    document.querySelectorAll('.toggle-group .toggle-btn').forEach(b => {
        if (b.parentElement === btn.parentElement) {
            b.classList.remove('active');
        }
    });
    btn.classList.add('active');
    updateURLParams();
}

function selectFFT(size, btn) {
    selectedFFT = size;
    document.querySelectorAll('.toggle-group .toggle-btn').forEach(b => {
        if (b.parentElement === btn.parentElement) {
            b.classList.remove('active');
        }
    });
    btn.classList.add('active');
    updateURLParams();
}

function incrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.min(parseInt(input.value) + 1, 30);
    updateURLParams();
}

function decrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.max(parseInt(input.value) - 1, 1);
    updateURLParams();
}

function incrementWarmup() {
    const input = document.getElementById('warmupRuns');
    input.value = Math.min(parseInt(input.value) + 1, 20);
    updateURLParams();
}

function decrementWarmup() {
    const input = document.getElementById('warmupRuns');
    input.value = Math.max(parseInt(input.value) - 1, 0);
    updateURLParams();
}

function incrementStableRuns() {
    const input = document.getElementById('stableRuns');
    input.value = Math.min(parseInt(input.value) + 1, 10);
    updateURLParams();
}

function decrementStableRuns() {
    const input = document.getElementById('stableRuns');
    input.value = Math.max(parseInt(input.value) - 1, 2);
    updateURLParams();
}

function toggleConfig() {
    const content = document.getElementById('configContent');
    const btn = document.getElementById('collapseBtn');
    content.classList.toggle('collapsed');
    btn.classList.toggle('collapsed');
}

function toggleAdvanced() {
    const content = document.getElementById('advancedContent');
    const btn = document.getElementById('advancedCollapseBtn');
    const isCollapsed = content.classList.toggle('collapsed');
    btn.textContent = isCollapsed ? '+' : '−';
    btn.classList.toggle('collapsed', !isCollapsed);
}

function toggleStableMode() {
    const checkbox = document.getElementById('stableMode');
    const container = document.getElementById('stableRunsContainer');

    if (checkbox.checked) {
        container.style.display = 'flex';
        container.classList.add('fade-in');
    } else {
        container.style.display = 'none';
    }
    updateURLParams();
}

function selectSignal(signalId) {
    selectedSignal = signalId;

    // Update UI
    document.querySelectorAll('.signal-pill').forEach(opt => {
        opt.classList.toggle('selected', opt.dataset.signal === signalId);
    });

    // Show/hide seed input
    const seedContainer = document.getElementById('seedContainer');
    const signalType = signalTypes.find(s => s.id === signalId);
    seedContainer.style.display = signalType?.has_seed ? 'flex' : 'none';
    updateURLParams();
}

function updateConfigSummary() {
    // No longer needed - removed config summary display
}

function renderSignalOptions() {
    const grid = document.getElementById('signalGrid');
    grid.innerHTML = signalTypes.map(signal => `
        <button class="signal-pill ${signal.id === selectedSignal ? 'selected' : ''} ${signal.recommended ? 'recommended' : ''}"
             data-signal="${signal.id}"
             onclick="selectSignal('${signal.id}')">
            <span class="pill-icon">${signal.icon}</span>
            <span>${signal.name}</span>
        </button>
    `).join('');
}

// Calculate frames for given duration and FFT size
function calculateFrames(durationSec, nFft, hopLength = HOP_LENGTH) {
    const samples = durationSec * SAMPLE_RATE;
    return Math.floor((samples - nFft) / hopLength) + 1;
}

// Format time with std dev
function formatTimeWithStd(avg, std) {
    if (std && std > 0.01) {
        return `${avg.toFixed(2)}ms ± ${std.toFixed(2)}ms`;
    }
    return `${avg.toFixed(2)}ms`;
}

// Save result to localStorage
function saveResult(result, config) {
    try {
        const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        history.unshift({
            timestamp: new Date().toISOString(),
            config: config,
            results: result
        });
        // Keep last 20 results
        if (history.length > 20) history.pop();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        console.warn('Could not save to localStorage:', e);
    }
}

// Load history from localStorage
function loadHistory() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch (e) {
        return [];
    }
}

// Main Benchmark Function
async function runBenchmark() {
    const stableMode = document.getElementById('stableMode').checked;
    const stableRuns = parseInt(document.getElementById('stableRuns').value);

    const config = {
        duration: selectedDuration,
        n_fft: selectedFFT,
        hop_length: HOP_LENGTH,
        n_mels: 80,
        iterations: parseInt(document.getElementById('runs').value),
        warmup_runs: parseInt(document.getElementById('warmupRuns').value),
        signal_type: selectedSignal,
        seed: parseInt(document.getElementById('seedInput').value) || 42,
        stable_mode: stableMode,
        stable_runs: stableRuns
    };

    // Show loading
    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('runBtn').disabled = true;

    // Update loading text for stable mode
    const loadingSubtext = document.getElementById('loadingSubtext');
    const loadingProgress = document.getElementById('loadingProgress');

    if (stableMode) {
        loadingSubtext.textContent = `Running ${stableRuns} complete passes for stable results`;
        loadingProgress.style.display = 'block';
    } else {
        loadingSubtext.textContent = 'This may take 30-60 seconds';
        loadingProgress.style.display = 'none';
    }

    try {
        const response = await fetch(`${API_BASE}/benchmark/both`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Benchmark failed');
        }

        const results = await response.json();
        benchmarkResults = results;

        // Save to history
        saveResult(results, config);

        // Display results
        displayResults(results, config);

    } catch (error) {
        console.error('Benchmark error:', error);
        alert('Benchmark failed. Make sure the backend is running!\n\nError: ' + error.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('runBtn').disabled = false;
    }
}

// Display Results
function displayResults(results, config) {
    const mojoIsFaster = results.mojo.avg_time_ms < results.librosa.avg_time_ms;
    const fasterPct = Math.abs(results.faster_percentage);

    // Show all sections
    document.getElementById('heroResult').style.display = 'block';
    document.getElementById('comparisonGrid').style.display = 'grid';
    document.getElementById('statsRow').style.display = 'flex';
    document.getElementById('chartCard').style.display = 'block';
    document.getElementById('actions').style.display = 'flex';

    // Show stable mode indicator
    const stableIndicator = document.getElementById('stableIndicator');
    if (results.stable_mode) {
        stableIndicator.style.display = 'flex';
        document.getElementById('stableInfo').textContent = `Median of ${results.num_runs} runs`;
    } else {
        stableIndicator.style.display = 'none';
    }

    // HERO SECTION
    document.getElementById('heroNumber').textContent = `${fasterPct.toFixed(1)}%`;

    const heroText = document.querySelector('.hero-text');
    const heroBadge = document.querySelector('.hero-badge');
    if (mojoIsFaster) {
        heroText.textContent = 'faster than librosa';
        heroBadge.textContent = 'mojo-audio wins';
        heroBadge.style.background = '#00ff9d';
        heroBadge.style.boxShadow = '0 4px 12px rgba(0, 255, 157, 0.4), 0 0 20px rgba(0, 255, 157, 0.3)';
    } else {
        heroText.textContent = 'slower than librosa';
        heroBadge.textContent = 'librosa wins';
        heroBadge.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
        heroBadge.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)';
    }

    // Winner badges
    const librosaBadge = document.getElementById('librosaBadge');
    const mojoBadge = document.getElementById('mojoBadge');
    const librosaCard = document.getElementById('librosaCard');
    const mojoCard = document.getElementById('mojoCard');

    librosaCard.classList.remove('winner');
    mojoCard.classList.remove('winner');
    librosaBadge.style.display = 'none';
    mojoBadge.style.display = 'none';

    if (mojoIsFaster) {
        mojoCard.classList.add('winner');
        mojoBadge.style.display = 'inline-block';
        mojoBadge.style.background = '#00ff9d';
        mojoBadge.style.boxShadow = '0 0 15px rgba(0, 255, 157, 0.3)';
    } else {
        librosaCard.classList.add('winner');
        librosaBadge.style.display = 'inline-block';
        librosaBadge.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
        librosaBadge.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.3)';
    }

    // Results with std dev
    document.getElementById('librosaTime').textContent =
        formatTimeWithStd(results.librosa.avg_time_ms, results.librosa.std_time_ms);
    document.getElementById('librosaThroughput').textContent =
        `${Math.round(results.librosa.throughput_realtime)}× realtime`;

    document.getElementById('mojoTime').textContent =
        formatTimeWithStd(results.mojo.avg_time_ms, results.mojo.std_time_ms);
    document.getElementById('mojoThroughput').textContent =
        `${Math.round(results.mojo.throughput_realtime)}× realtime`;

    // Progress bars
    if (mojoIsFaster) {
        document.getElementById('mojoBar').style.width = '100%';
        document.getElementById('librosaBar').style.width =
            `${(results.mojo.avg_time_ms / results.librosa.avg_time_ms) * 100}%`;
    } else {
        document.getElementById('librosaBar').style.width = '100%';
        document.getElementById('mojoBar').style.width =
            `${(results.librosa.avg_time_ms / results.mojo.avg_time_ms) * 100}%`;
    }

    // Stats with dynamic frame count
    const frames = calculateFrames(config.duration, config.n_fft);
    document.getElementById('speedupFactor').textContent =
        `${Math.abs(results.speedup_factor).toFixed(2)}×`;
    document.getElementById('framesProcessed').textContent =
        frames.toLocaleString();
    document.getElementById('runsAveraged').textContent =
        results.stable_mode ? `${results.num_runs}×${config.iterations}` : config.iterations;

    // Create chart
    createOptimizationChart();

    // Scroll to results
    setTimeout(() => {
        document.getElementById('heroResult').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }, 300);
}

// Create Optimization Chart with Dev Coffee brand colors
function createOptimizationChart() {
    const ctx = document.getElementById('optimizationChart');

    if (optimizationChart) {
        optimizationChart.destroy();
    }

    // Updated optimization journey with current benchmarks (30s audio, n_fft=400)
    const optimizations = [
        { name: 'Naive Python', time: 476, desc: 'Initial recursive FFT' },
        { name: 'Iterative FFT', time: 165, desc: 'Cooley-Tukey DIT' },
        { name: '+Twiddles', time: 97, desc: 'Precomputed twiddle factors' },
        { name: '+Float32', time: 78, desc: '2x SIMD width' },
        { name: '+SoA Layout', time: 55, desc: 'Structure of Arrays' },
        { name: '+SIMD FFT', time: 42, desc: 'Vectorized butterflies' },
        { name: '+Radix-4', time: 35, desc: '4-point butterflies' },
        { name: '+Split-Radix', time: 28, desc: 'Hybrid radix-2/4' },
        { name: '+O3', time: 25, desc: 'Compiler optimizations' }
    ];

    // Get current librosa baseline from results if available
    const librosaBaseline = benchmarkResults ?
        benchmarkResults.librosa.avg_time_ms : 35;

    optimizationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: optimizations.map(o => o.name),
            datasets: [
                {
                    label: 'mojo-audio (ms)',
                    data: optimizations.map(o => o.time),
                    borderColor: '#00ff9d',
                    backgroundColor: 'rgba(0, 255, 157, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#00ff9d',
                    pointBorderColor: '#0d1117',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: `librosa baseline (${librosaBaseline.toFixed(0)}ms)`,
                    data: new Array(optimizations.length).fill(librosaBaseline),
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    borderDash: [8, 4],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 13, weight: '500', family: "'SF Mono', monospace" },
                        color: '#8b949e',
                        usePointStyle: true,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(22, 27, 34, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#8b949e',
                    borderColor: 'rgba(0, 255, 157, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                const opt = optimizations[context.dataIndex];
                                return [`${context.parsed.y.toFixed(1)}ms`, opt.desc];
                            }
                            return `librosa: ${librosaBaseline.toFixed(1)}ms`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    ticks: {
                        font: { size: 12, family: "'SF Mono', monospace" },
                        color: '#8b949e',
                        callback: value => value + 'ms'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 11, family: "'SF Mono', monospace" },
                        color: '#8b949e',
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Download Results
function downloadResults() {
    if (!benchmarkResults) return;

    const data = {
        timestamp: new Date().toISOString(),
        configuration: {
            duration: selectedDuration,
            fft_size: selectedFFT,
            hop_length: HOP_LENGTH,
            n_mels: 80,
            iterations: parseInt(document.getElementById('runs').value),
            warmup_runs: parseInt(document.getElementById('warmupRuns').value),
            signal_type: selectedSignal,
            stable_mode: document.getElementById('stableMode').checked,
            stable_runs: parseInt(document.getElementById('stableRuns').value)
        },
        results: benchmarkResults,
        frames_processed: calculateFrames(selectedDuration, selectedFFT)
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mojo-audio-benchmark-${Date.now()}.json`;
    a.click();
}

// Fetch signal types from API
async function fetchSignalTypes() {
    try {
        const response = await fetch(`${API_BASE}/signal-types`);
        if (response.ok) {
            const data = await response.json();
            signalTypes = data.types;
        }
    } catch (e) {
        console.warn('Could not fetch signal types, using defaults');
    }
    renderSignalOptions();
}

// Show last result on page load if available
function showLastResult() {
    const history = loadHistory();
    if (history.length > 0) {
        const last = history[0];
        console.log('Last benchmark:', last.timestamp);
        // Could auto-display last result here if desired
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('mojo-audio benchmark UI loaded');

    // Load settings from URL params first
    loadFromURLParams();

    // Render signal options (uses selectedSignal from URL)
    renderSignalOptions();

    // Show/hide seed input based on selected signal
    const signalType = signalTypes.find(s => s.id === selectedSignal);
    document.getElementById('seedContainer').style.display = signalType?.has_seed ? 'flex' : 'none';

    // Try to fetch signal types from API
    fetchSignalTypes();

    // Seed input change handler
    document.getElementById('seedInput').addEventListener('change', updateURLParams);

    // Load history
    showLastResult();
});
