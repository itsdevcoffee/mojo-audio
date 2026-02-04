/**
 * Demo Mode for mojo-audio Benchmark UI
 *
 * Features:
 * - Pre-computed benchmark data (24 configs)
 * - Smart interpolation for run count variations
 * - Realistic variance to simulate real benchmarks
 * - Simulated loading time based on config complexity
 */

// Pre-loaded benchmark data (loaded from JSON)
let BENCHMARK_DATA = null;
let CURRENT_MACHINE_DATA = null;

/**
 * Load pre-computed benchmark data for a specific machine
 */
async function loadBenchmarkData(machineId = null) {
    try {
        // Determine which data file to load
        let dataFile = '/static/data/benchmark_results.json';  // Default

        if (machineId && window.machinesConfig) {
            const machine = window.machinesConfig.machines[machineId];
            if (machine && machine.data_file) {
                dataFile = `/static/data/${machine.data_file}`;
            }
        }

        const response = await fetch(dataFile);
        const data = await response.json();

        // Check if data has metadata wrapper
        if (data.benchmarks) {
            BENCHMARK_DATA = data.benchmarks;
            CURRENT_MACHINE_DATA = data;
        } else {
            BENCHMARK_DATA = data;
            CURRENT_MACHINE_DATA = { benchmarks: data };
        }

        console.log(`✅ Loaded benchmark data for ${machineId || 'default'}:`, Object.keys(BENCHMARK_DATA).length, 'configs');
        return true;
    } catch (error) {
        console.error('❌ Failed to load benchmark data:', error);
        return false;
    }
}

/**
 * Get config key for lookup
 */
function getConfigKey(duration, fftSize, blas) {
    return `${duration}s_${fftSize}fft_${blas}`;
}

/**
 * Add realistic variance to simulate benchmark variability
 * Real benchmarks have ~3-8% variance
 */
function addRealisticVariance(value, baseVariance = 0.05) {
    // Random variance between -baseVariance and +baseVariance
    const variance = (Math.random() - 0.5) * 2 * baseVariance;
    return value * (1 + variance);
}

/**
 * Interpolate results based on run count
 * More runs = more accurate (lower std deviation)
 */
function interpolateByRunCount(baseResult, targetRuns, baseRuns = 20) {
    // Runs don't affect avg time much, but affect std deviation
    // More runs = more stable (lower std)
    const stdScale = Math.sqrt(baseRuns / targetRuns);

    // Add small variance to avg time (simulates different runs)
    const avgTime = addRealisticVariance(baseResult.avg_time_ms, 0.03);
    const stdTime = baseResult.std_time_ms * stdScale;

    return {
        avg_time_ms: parseFloat(avgTime.toFixed(2)),
        std_time_ms: parseFloat(stdTime.toFixed(2)),
        throughput_realtime: parseFloat((1000 * baseResult.throughput_realtime / avgTime * baseResult.avg_time_ms).toFixed(1))
    };
}

/**
 * Get demo benchmark results with interpolation and variance
 */
function getDemoResults(config) {
    if (!BENCHMARK_DATA) {
        throw new Error('Benchmark data not loaded');
    }

    const configKey = getConfigKey(config.duration, config.n_fft, config.blas_backend);
    const baseData = BENCHMARK_DATA[configKey];

    if (!baseData) {
        console.warn(`No data for ${configKey}, using fallback`);
        // Fallback to 10s_400fft_openblas and scale
        const fallback = BENCHMARK_DATA['10s_400fft_openblas'];
        return interpolateFallback(fallback, config);
    }

    // Interpolate based on run count
    const mojo = interpolateByRunCount(baseData.mojo, config.iterations);
    const librosa = interpolateByRunCount(baseData.librosa, config.iterations);

    // Recalculate speedup with interpolated values
    const speedup = librosa.avg_time_ms / mojo.avg_time_ms;
    const fasterPct = ((librosa.avg_time_ms - mojo.avg_time_ms) / librosa.avg_time_ms) * 100;

    return {
        mojo: {
            implementation: "mojo-audio",
            duration: config.duration,
            avg_time_ms: mojo.avg_time_ms,
            std_time_ms: mojo.std_time_ms,
            throughput_realtime: mojo.throughput_realtime,
            iterations: config.iterations,
            success: true
        },
        librosa: {
            implementation: `librosa (${config.blas_backend.toUpperCase()})`,
            duration: config.duration,
            avg_time_ms: librosa.avg_time_ms,
            std_time_ms: librosa.std_time_ms,
            throughput_realtime: librosa.throughput_realtime,
            iterations: config.iterations,
            success: true
        },
        speedup_factor: parseFloat(speedup.toFixed(2)),
        faster_percentage: parseFloat(fasterPct.toFixed(1)),
        mojo_is_faster: mojo.avg_time_ms < librosa.avg_time_ms
    };
}

/**
 * Fallback interpolation if exact config not found
 */
function interpolateFallback(baseData, targetConfig) {
    const durationScale = targetConfig.duration / baseData.config.duration;
    const fftScale = (targetConfig.n_fft * Math.log2(targetConfig.n_fft)) /
                     (baseData.config.fft_size * Math.log2(baseData.config.fft_size));

    const scale = durationScale * fftScale;

    const mojoScaled = {
        avg_time_ms: baseData.mojo.avg_time_ms * scale,
        std_time_ms: baseData.mojo.std_time_ms * scale,
        throughput_realtime: baseData.mojo.throughput_realtime / scale
    };

    const librosaScaled = {
        avg_time_ms: baseData.librosa.avg_time_ms * scale,
        std_time_ms: baseData.librosa.std_time_ms * scale,
        throughput_realtime: baseData.librosa.throughput_realtime / scale
    };

    // Apply run count interpolation
    const mojo = interpolateByRunCount(mojoScaled, targetConfig.iterations);
    const librosa = interpolateByRunCount(librosaScaled, targetConfig.iterations);

    const speedup = librosa.avg_time_ms / mojo.avg_time_ms;
    const fasterPct = ((librosa.avg_time_ms - mojo.avg_time_ms) / librosa.avg_time_ms) * 100;

    return {
        mojo: {
            implementation: "mojo-audio",
            duration: targetConfig.duration,
            avg_time_ms: mojo.avg_time_ms,
            std_time_ms: mojo.std_time_ms,
            throughput_realtime: mojo.throughput_realtime,
            iterations: targetConfig.iterations,
            success: true
        },
        librosa: {
            implementation: `librosa (${targetConfig.blas_backend.toUpperCase()})`,
            duration: targetConfig.duration,
            avg_time_ms: librosa.avg_time_ms,
            std_time_ms: librosa.std_time_ms,
            throughput_realtime: librosa.throughput_realtime,
            iterations: targetConfig.iterations,
            success: true
        },
        speedup_factor: parseFloat(speedup.toFixed(2)),
        faster_percentage: parseFloat(fasterPct.toFixed(1)),
        mojo_is_faster: mojo.avg_time_ms < librosa.avg_time_ms
    };
}

/**
 * Calculate realistic loading time based on config
 * Simulates actual benchmark execution time (shortened for demo)
 */
function calculateLoadingTime(config) {
    // Base time: 2 seconds minimum for realism
    let baseTime = 2000;

    // Duration factor (linear scaling, but capped)
    const durationFactor = Math.min(config.duration / 10, 3); // Max 3x

    // FFT size factor (logarithmic scaling)
    const fftFactor = Math.log2(config.n_fft / 400) + 1; // 1x at 400, ~2x at 1024

    // Iterations factor (more runs = longer, but not linear)
    const iterationsFactor = Math.log10(config.iterations) + 1; // 1x at 1, ~2x at 100

    // BLAS factor (MKL slightly faster than OpenBLAS in simulation)
    const blasFactor = config.blas_backend === 'mkl' ? 0.9 : 1.0;

    // Total time (with some randomness for realism)
    const totalTime = baseTime * durationFactor * fftFactor * iterationsFactor * blasFactor;

    // Add ±10% random variance
    const withVariance = addRealisticVariance(totalTime, 0.1);

    // Clamp between 1.5s and 8s for UX
    return Math.max(1500, Math.min(8000, withVariance));
}

/**
 * Run demo benchmark with simulated loading
 */
async function runDemoBenchmark(config, onProgress) {
    const loadingTime = calculateLoadingTime(config);

    console.log(`🎮 Demo benchmark: ${loadingTime.toFixed(0)}ms loading time`);

    // Simulate progress updates
    const progressSteps = 10;
    const stepTime = loadingTime / progressSteps;

    for (let i = 0; i <= progressSteps; i++) {
        if (onProgress) {
            onProgress({
                progress: i / progressSteps,
                message: i < progressSteps / 2
                    ? `Running mojo-audio benchmark... (${i}/${progressSteps})`
                    : `Running librosa benchmark... (${i}/${progressSteps})`
            });
        }

        await new Promise(resolve => setTimeout(resolve, stepTime));
    }

    // Return results
    return getDemoResults(config);
}

/**
 * Initialize demo mode
 */
async function initDemoMode() {
    const loaded = await loadBenchmarkData();
    if (!loaded) {
        console.error('Failed to initialize demo mode');
        return false;
    }

    console.log('🎮 Demo mode initialized');
    return true;
}

// Export for use in main.js
window.DemoMode = {
    init: initDemoMode,
    runBenchmark: runDemoBenchmark,
    isReady: () => BENCHMARK_DATA !== null
};
