// mojo-audio Benchmark UI - JavaScript
// Raycast-inspired smooth interactions

const API_BASE = 'http://localhost:8000/api';

// UI State
let benchmarkResults = null;
let optimizationChart = null;

// Helper Functions
function incrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.min(parseInt(input.value) + 1, 20);
}

function decrementRuns() {
    const input = document.getElementById('runs');
    input.value = Math.max(parseInt(input.value) - 1, 1);
}

function getSelectedDuration() {
    return parseInt(document.querySelector('input[name="duration"]:checked').value);
}

function getSelectedFFTSize() {
    return parseInt(document.querySelector('input[name="fft_size"]:checked').value);
}

function getIterations() {
    return parseInt(document.getElementById('runs').value);
}

// Main Benchmark Function
async function runBenchmark() {
    const config = {
        duration: getSelectedDuration(),
        n_fft: getSelectedFFTSize(),
        iterations: getIterations()
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

        // Display results with animation
        displayResults(results, config);

    } catch (error) {
        console.error('Benchmark error:', error);
        alert('Benchmark failed. Make sure the backend is running!');
    } finally {
        // Hide loading
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('runBtn').disabled = false;
    }
}

// Display Results
function displayResults(results, config) {
    const resultsCard = document.getElementById('resultsCard');
    const chartCard = document.getElementById('chartCard');
    const actions = document.getElementById('actions');

    // Show cards with fade-in
    resultsCard.style.display = 'block';
    chartCard.style.display = 'block';
    actions.style.display = 'flex';

    resultsCard.classList.add('fade-in');
    chartCard.classList.add('fade-in');

    // librosa results
    document.getElementById('librosaTime').textContent =
        `${results.librosa.avg_time_ms.toFixed(2)}ms`;
    document.getElementById('librosaThroughput').textContent =
        `${Math.round(results.librosa.throughput_realtime)}x realtime`;

    // mojo-audio results
    document.getElementById('mojoTime').textContent =
        `${results.mojo.avg_time_ms.toFixed(2)}ms`;
    document.getElementById('mojoThroughput').textContent =
        `${Math.round(results.mojo.throughput_realtime)}x realtime`;

    // Animate progress bars
    const maxTime = Math.max(results.librosa.avg_time_ms, results.mojo.avg_time_ms);
    setTimeout(() => {
        document.getElementById('librosaBar').style.width =
            `${(results.librosa.avg_time_ms / maxTime) * 100}%`;
        document.getElementById('mojoBar').style.width =
            `${(results.mojo.avg_time_ms / maxTime) * 100}%`;
    }, 100);

    // Stats
    document.getElementById('speedupFactor').textContent =
        `${results.speedup_factor}x`;
    document.getElementById('fasterPct').textContent =
        `${results.faster_percentage}%`;
    document.getElementById('framesProcessed').textContent =
        '~2,998';

    // Success badge
    const badge = document.getElementById('successBadge');
    if (results.mojo_is_faster) {
        badge.style.display = 'flex';
        document.getElementById('badgeText').textContent =
            `mojo-audio is ${results.faster_percentage}% faster!`;
    }

    // Create optimization chart
    createOptimizationChart();

    // Smooth scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

// Create Optimization Journey Chart
function createOptimizationChart() {
    const ctx = document.getElementById('optimizationChart');

    if (optimizationChart) {
        optimizationChart.destroy();
    }

    const optimizations = [
        { name: 'Naive', time: 476 },
        { name: 'Iterative FFT', time: 165 },
        { name: '+ Twiddles', time: 97 },
        { name: '+ Sparse', time: 78 },
        { name: '+ Caching', time: 38 },
        { name: '+ Float32', time: 34.4 },
        { name: '+ True RFFT', time: 24 },
        { name: '+ Parallel', time: 18 },
        { name: '+ O3', time: 12 }
    ];

    optimizationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: optimizations.map(o => o.name),
            datasets: [
                {
                    label: 'Processing Time (ms)',
                    data: optimizations.map(o => o.time),
                    borderColor: '#FF8A5B',
                    backgroundColor: 'rgba(255, 138, 91, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#FF6B35',
                    pointBorderColor: '#FFFFFF',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'librosa target',
                    data: new Array(optimizations.length).fill(15),
                    borderColor: '#60A5FA',
                    borderWidth: 2,
                    borderDash: [5, 5],
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
                            weight: '500'
                        },
                        color: '#6B6B6B',
                        usePointStyle: true,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#1F1F1F',
                    bodyColor: '#6B6B6B',
                    borderColor: '#E5E5E5',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    titleFont: {
                        size: 14,
                        weight: '600'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.y.toFixed(2)}ms`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#F3F4F6',
                        drawBorder: false
                    },
                    ticks: {
                        font: {
                            size: 12,
                            family: "'SF Mono', monospace"
                        },
                        color: '#9CA3AF',
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
                            size: 11
                        },
                        color: '#9CA3AF',
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
});
