/**
 * Configuration for mojo-audio Benchmark UI
 *
 * Controls whether to use demo mode (static) or live server mode
 */

// Auto-detect mode based on environment
function detectMode() {
    // Check if we're running on a static host (Vercel, Netlify, GitHub Pages, etc.)
    const isStaticHost =
        window.location.hostname.includes('vercel.app') ||
        window.location.hostname.includes('netlify.app') ||
        window.location.hostname.includes('github.io') ||
        window.location.hostname.includes('pages.dev'); // Cloudflare Pages

    // Check if backend is available (for local development with server)
    const hasBackend = window.location.port === '8000' ||
                       window.location.pathname.includes('/api/');

    // Default to demo mode for static hosts, server mode for local
    return isStaticHost ? 'demo' : (hasBackend ? 'server' : 'demo');
}

// Configuration object
const AppConfig = {
    // Mode: 'demo' or 'server' or 'auto'
    // - 'demo': Use pre-computed data (static deployment)
    // - 'server': Use live FastAPI backend (requires server running)
    // - 'auto': Auto-detect based on environment
    mode: 'auto',

    // API endpoints (only used in server mode)
    api: {
        base: window.location.origin,
        endpoints: {
            health: '/api/health',
            benchmarkMojo: '/api/benchmark/mojo',
            benchmarkLibrosa: '/api/benchmark/librosa',
            benchmarkBoth: '/api/benchmark/both'
        }
    },

    // Demo mode configuration
    demo: {
        dataPath: '/static/data/benchmark_results.json',
        showBanner: true, // Show "Demo Mode" indicator
        enableVariance: true, // Add realistic variance to results
        variancePercent: 0.05, // ±5% variance
        minLoadingTime: 1500, // Minimum loading time (ms)
        maxLoadingTime: 8000  // Maximum loading time (ms)
    },

    // Server mode configuration
    server: {
        timeout: 120000, // 2 minutes
        retryAttempts: 1,
        retryDelay: 1000
    }
};

/**
 * Get the active mode
 */
function getActiveMode() {
    if (AppConfig.mode === 'auto') {
        return detectMode();
    }
    return AppConfig.mode;
}

/**
 * Check if demo mode is active
 */
function isDemoMode() {
    return getActiveMode() === 'demo';
}

/**
 * Check if server mode is active
 */
function isServerMode() {
    return getActiveMode() === 'server';
}

/**
 * Verify backend availability (only in server mode)
 */
async function verifyBackend() {
    if (isDemoMode()) {
        return false;
    }

    try {
        const response = await fetch(AppConfig.api.endpoints.health, {
            timeout: 5000
        });
        return response.ok;
    } catch (error) {
        console.warn('Backend not available:', error.message);
        return false;
    }
}

/**
 * Initialize app configuration
 */
async function initializeConfig() {
    const mode = getActiveMode();

    console.log('🔧 Configuration:');
    console.log('   Mode:', mode);
    console.log('   Host:', window.location.hostname);

    if (mode === 'server') {
        const backendAvailable = await verifyBackend();
        console.log('   Backend:', backendAvailable ? '✅ Available' : '❌ Not available');

        if (!backendAvailable) {
            console.warn('⚠️  Falling back to demo mode (backend not available)');
            AppConfig.mode = 'demo';
            return 'demo';
        }
    }

    return mode;
}

// Export configuration
window.AppConfig = AppConfig;
window.getActiveMode = getActiveMode;
window.isDemoMode = isDemoMode;
window.isServerMode = isServerMode;
window.initializeConfig = initializeConfig;
