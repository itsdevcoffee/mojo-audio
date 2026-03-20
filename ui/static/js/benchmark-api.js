/**
 * Unified Benchmark API
 *
 * Handles both demo mode (static) and server mode (live API) transparently
 */

/**
 * Run benchmark with automatic mode selection
 */
async function runBenchmark(config, options = {}) {
    const mode = getActiveMode();

    console.log(`🎯 Running benchmark in ${mode} mode`);
    console.log('   Config:', config);

    if (mode === 'demo') {
        return await runDemoBenchmark(config, options);
    } else {
        return await runServerBenchmark(config, options);
    }
}

/**
 * Run benchmark using demo mode (static data)
 */
async function runDemoBenchmark(config, options = {}) {
    const { onProgress } = options;

    // Check if demo mode is initialized
    if (!window.DemoMode || !window.DemoMode.isReady()) {
        throw new Error('Demo mode not initialized. Call DemoMode.init() first.');
    }

    // Run demo benchmark with simulated loading
    return await window.DemoMode.runBenchmark(config, onProgress);
}

/**
 * Run benchmark using server mode (live API)
 */
async function runServerBenchmark(config, options = {}) {
    const { onProgress } = options;

    const body = JSON.stringify({
        duration: config.duration,
        n_fft: config.n_fft,
        hop_length: config.hop_length || 160,
        n_mels: config.n_mels || 80,
        iterations: config.iterations,
        blas_backend: config.blas_backend
    });

    const fetchOpts = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: AbortSignal.timeout(AppConfig.server.timeout)
    };

    try {
        // Phase 1: Run mojo-audio
        if (onProgress) {
            onProgress({ progress: 0.1, message: 'Running mojo-audio benchmark...' });
        }

        const mojoRes = await fetch(AppConfig.api.endpoints.benchmarkMojo, fetchOpts);
        if (!mojoRes.ok) {
            const error = await mojoRes.json();
            throw new Error(error.detail || 'Mojo benchmark failed');
        }
        const mojo = await mojoRes.json();

        // Phase 2: Run librosa
        if (onProgress) {
            onProgress({ progress: 0.5, message: 'Running librosa benchmark...' });
        }

        const librosaRes = await fetch(AppConfig.api.endpoints.benchmarkLibrosa, fetchOpts);
        if (!librosaRes.ok) {
            const error = await librosaRes.json();
            throw new Error(error.detail || 'Librosa benchmark failed');
        }
        const librosa = await librosaRes.json();

        // Compute comparison
        const speedup_factor = librosa.avg_time_ms / mojo.avg_time_ms;
        const faster_percentage = ((librosa.avg_time_ms - mojo.avg_time_ms) / librosa.avg_time_ms) * 100;

        if (onProgress) {
            onProgress({ progress: 1, message: 'Benchmark complete!' });
        }

        return {
            mojo,
            librosa,
            speedup_factor: Math.round(speedup_factor * 100) / 100,
            faster_percentage: Math.round(faster_percentage * 10) / 10,
            mojo_is_faster: mojo.avg_time_ms < librosa.avg_time_ms
        };

    } catch (error) {
        if (error.name === 'TimeoutError' || error.name === 'AbortError') {
            throw new Error('Benchmark timeout - try reducing duration or iterations');
        }

        // If server fails, optionally fall back to demo mode
        if (AppConfig.demo.fallbackOnError) {
            console.warn('Server benchmark failed, falling back to demo mode:', error);
            return await runDemoBenchmark(config, options);
        }

        throw error;
    }
}

/**
 * Health check
 */
async function checkHealth() {
    if (isDemoMode()) {
        return {
            status: 'demo',
            message: 'Running in demo mode with pre-computed data'
        };
    }

    try {
        const response = await fetch(AppConfig.api.endpoints.health);
        if (!response.ok) {
            throw new Error('Health check failed');
        }
        return await response.json();
    } catch (error) {
        return {
            status: 'error',
            message: error.message
        };
    }
}

/**
 * Initialize benchmark API
 */
async function initializeBenchmarkAPI() {
    // Initialize config first
    const mode = await initializeConfig();

    console.log(`🚀 Initializing benchmark API (${mode} mode)...`);

    if (mode === 'demo') {
        // Initialize demo mode
        if (window.DemoMode) {
            const ready = await window.DemoMode.init();
            if (!ready) {
                throw new Error('Failed to initialize demo mode');
            }
            console.log('✅ Demo mode initialized');
        } else {
            throw new Error('Demo mode script not loaded');
        }
    } else {
        // Verify server is available
        const health = await checkHealth();
        if (health.status === 'error') {
            console.warn('⚠️  Server health check failed:', health.message);
            console.warn('   Consider falling back to demo mode');
        } else {
            console.log('✅ Server mode initialized');
        }
    }

    return mode;
}

// Export API
window.BenchmarkAPI = {
    run: runBenchmark,
    health: checkHealth,
    init: initializeBenchmarkAPI,
    getMode: getActiveMode
};
