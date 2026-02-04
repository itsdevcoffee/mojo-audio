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

    // Show initial progress
    if (onProgress) {
        onProgress({ progress: 0, message: 'Starting benchmark...' });
    }

    try {
        const response = await fetch(AppConfig.api.endpoints.benchmarkBoth, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                duration: config.duration,
                n_fft: config.n_fft,
                hop_length: config.hop_length || 160,
                n_mels: config.n_mels || 80,
                iterations: config.iterations,
                blas_backend: config.blas_backend
            }),
            signal: AbortSignal.timeout(AppConfig.server.timeout)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Benchmark failed');
        }

        // Update progress to complete
        if (onProgress) {
            onProgress({ progress: 1, message: 'Benchmark complete!' });
        }

        return await response.json();

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
