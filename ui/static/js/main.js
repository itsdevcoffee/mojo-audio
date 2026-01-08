// mojo-audio Benchmark UI - Glassmorphic Enhanced
// Implements Gemini's UX improvements in Vanilla JS

const API_BASE = 'http://localhost:8000/api';

// UI State
let selectedDuration = 30;
let selectedFFT = 400;
let benchmarkResults = null;
let optimizationChart = null;

// Toggle Functions
function selectDuration(duration, btn) {
    selectedDuration = duration;
    // Update active state
    document.querySelectorAll('.toggle-group .toggle-btn').forEach(b => {
        if (b.parentElement === btn.parentElement) {
            b.classList.remove('active');
        }
    });
    btn.classList.add('active');
}

function selectFFT(size, btn) {
    selectedFFT = size;
    // Update active state
    document.querySelectorAll('.toggle-group .toggle-btn').forEach(b => {
        if (b.parentElement === btn.parentElement) {
            b.classList.remove('active');
        }
    });
    btn.classList.add('active');
}

function incrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.min(parseInt(input.value) + 1, 30);
}

function decrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.max(parseInt(input.value) - 1, 1);
}

function toggleConfig() {
    const content = document.getElementById('configContent');
    const btn = document.getElementById('collapseBtn');

    content.classList.toggle('collapsed');
    btn.classList.toggle('collapsed');
}

// Main Benchmark Function
async function runBenchmark() {
    const config = {
        duration: selectedDuration,
        n_fft: selectedFFT,
        iterations: parseInt(document.getElementById('runs').value)
    };

    // Show loading
    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('runBtn').disabled = true;

    try {
        // Call API
        const response = await fetch(`${API_BASE}/benchmark/both`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Benchmark failed');
        }

        const results = await response.json();
        benchmarkResults = results;

        // Display results with Gemini's improved UX
        displayResults(results, config);

        // Auto-collapse config (optional)
        // toggleConfig();

    } catch (error) {
        console.error('Benchmark error:', error);
        alert('Benchmark failed. Make sure the backend is running!\n\nError: ' + error.message);
    } finally {
        // Hide loading
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('runBtn').disabled = false;
    }
}

// Display Results (Gemini's improved layout - handles both win/loss!)
function displayResults(results, config) {
    // Determine actual winner
    const mojoIsFaster = results.mojo.avg_time_ms < results.librosa.avg_time_ms;
    const fasterPct = Math.abs(results.faster_percentage);
    const winner = mojoIsFaster ? 'mojo-audio' : 'librosa';

    // Show all sections
    document.getElementById('heroResult').style.display = 'block';
    document.getElementById('comparisonGrid').style.display = 'grid';
    document.getElementById('statsRow').style.display = 'flex';
    document.getElementById('chartCard').style.display = 'block';
    document.getElementById('actions').style.display = 'flex';

    // HERO SECTION (dynamic based on winner!)
    document.getElementById('heroNumber').textContent = `${fasterPct.toFixed(1)}%`;

    // Update hero text based on winner
    const heroText = document.querySelector('.hero-text');
    const heroBadge = document.querySelector('.hero-badge');
    if (mojoIsFaster) {
        heroText.textContent = 'faster than librosa';
        heroBadge.textContent = 'mojo-audio wins';
        heroBadge.style.background = 'linear-gradient(135deg, #f97316, #ea580c)';
    } else {
        heroText.textContent = 'slower than librosa';
        heroBadge.textContent = 'librosa wins';
        heroBadge.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
    }

    // Update card-level winner styling and badges
    const librosaBadge = document.getElementById('librosaBadge');
    const mojoBadge = document.getElementById('mojoBadge');
    const librosaCard = document.getElementById('librosaCard');
    const mojoCard = document.getElementById('mojoCard');

    // Reset both cards
    librosaCard.classList.remove('winner');
    mojoCard.classList.remove('winner');
    librosaBadge.style.display = 'none';
    mojoBadge.style.display = 'none';

    // Style the winner
    if (mojoIsFaster) {
        mojoCard.classList.add('winner');
        mojoBadge.style.display = 'inline-block';
        mojoBadge.style.background = 'linear-gradient(135deg, #f97316, #ea580c)';
    } else {
        librosaCard.classList.add('winner');
        librosaBadge.style.display = 'inline-block';
        librosaBadge.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
    }

    // librosa results
    document.getElementById('librosaTime').textContent =
        `${results.librosa.avg_time_ms.toFixed(2)}ms`;
    document.getElementById('librosaThroughput').textContent =
        `${Math.round(results.librosa.throughput_realtime)}× realtime`;

    // mojo-audio results
    document.getElementById('mojoTime').textContent =
        `${results.mojo.avg_time_ms.toFixed(2)}ms`;
    document.getElementById('mojoThroughput').textContent =
        `${Math.round(results.mojo.throughput_realtime)}× realtime`;

    // Progress bars (winner at 100%, loser scaled)
    if (mojoIsFaster) {
        // mojo wins
        document.getElementById('mojoBar').style.width = '100%';
        document.getElementById('librosaBar').style.width =
            `${(results.mojo.avg_time_ms / results.librosa.avg_time_ms) * 100}%`;
    } else {
        // librosa wins
        document.getElementById('librosaBar').style.width = '100%';
        document.getElementById('mojoBar').style.width =
            `${(results.librosa.avg_time_ms / results.mojo.avg_time_ms) * 100}%`;
    }

    // Stats
    document.getElementById('speedupFactor').textContent =
        `${Math.abs(results.speedup_factor).toFixed(2)}×`;
    document.getElementById('framesProcessed').textContent =
        '~2,998';
    document.getElementById('runsAveraged').textContent =
        config.iterations;

    // Create chart
    createOptimizationChart();

    // Smooth scroll to hero result
    setTimeout(() => {
        document.getElementById('heroResult').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }, 300);
}

// Create Optimization Chart (enhanced with librosa baseline!)
function createOptimizationChart() {
    const ctx = document.getElementById('optimizationChart');

    if (optimizationChart) {
        optimizationChart.destroy();
    }

    const optimizations = [
        { name: 'Naive', time: 476 },
        { name: 'Iterative FFT', time: 165 },
        { name: '+Twiddles', time: 97 },
        { name: '+Sparse', time: 78 },
        { name: '+Caching', time: 38 },
        { name: '+Float32', time: 34.4 },
        { name: '+RFFT', time: 24 },
        { name: '+Parallel', time: 18 },
        { name: '+O3', time: 12 }
    ];

    optimizationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: optimizations.map(o => o.name),
            datasets: [
                {
                    label: 'mojo-audio (ms)',
                    data: optimizations.map(o => o.time),
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#f97316',
                    pointBorderColor: '#FFFFFF',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'librosa baseline',
                    data: new Array(optimizations.length).fill(15),
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
                        font: {
                            size: 13,
                            weight: '500',
                            family: "'SF Mono', monospace"
                        },
                        color: '#666',
                        usePointStyle: true,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#1a1a1a',
                    bodyColor: '#666',
                    borderColor: '#e5e5e5',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    titleFont: {
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return `${context.parsed.y.toFixed(2)}ms`;
                            }
                            return 'librosa target: 15ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 12,
                            family: "'SF Mono', monospace"
                        },
                        color: '#999',
                        callback: function(value) {
                            return value + 'ms';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11,
                            family: "'SF Mono', monospace"
                        },
                        color: '#999',
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
            iterations: parseInt(document.getElementById('runs').value)
        },
        results: benchmarkResults
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mojo-audio-benchmark-${Date.now()}.json`;
    a.click();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('mojo-audio benchmark UI loaded! 🔥');
    console.log('Glassmorphic design with Gemini UX improvements');
});
