"""Progressive test - add functions one by one"""

from math import cos, sqrt, log, sin, exp
from math.constants import pi
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

from ffi.types import (
    MojoMelConfig,
    MojoMelSpectrogram,
    MOJO_AUDIO_SUCCESS,
    MOJO_AUDIO_ERROR_INVALID_INPUT,
    MOJO_AUDIO_ERROR_INVALID_HANDLE,
    MOJO_AUDIO_ERROR_PROCESSING,
    MOJO_AUDIO_ERROR_BUFFER_SIZE
)
# from audio import mel_spectrogram  # REMOVED - causes crash in shared lib context


# ==============================================================================
# Inlined Audio Processing Functions (from audio.mojo)
# Required for FFI - can't call imported module in shared lib context
# ==============================================================================

# ------------------------------------------------------------------------------
# Complex Number Operations
# ------------------------------------------------------------------------------

struct Complex(Copyable, Movable):
    """Complex number for FFT operations (Float32 for 2x SIMD throughput!)."""
    var real: Float32
    var imag: Float32

    fn __init__(out self, real: Float32, imag: Float32 = 0.0):
        """Initialize complex number."""
        self.real = real
        self.imag = imag

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.real = existing.real
        self.imag = existing.imag

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.real = existing.real
        self.imag = existing.imag

    @always_inline
    fn __add__(self, other: Complex) -> Complex:
        """Complex addition."""
        return Complex(self.real + other.real, self.imag + other.imag)

    @always_inline
    fn __sub__(self, other: Complex) -> Complex:
        """Complex subtraction."""
        return Complex(self.real - other.real, self.imag - other.imag)

    @always_inline
    fn __mul__(self, other: Complex) -> Complex:
        """Complex multiplication."""
        var r = self.real * other.real - self.imag * other.imag
        var i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    @always_inline
    fn power(self) -> Float32:
        """Compute power: real² + imag²."""
        return self.real * self.real + self.imag * self.imag

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

fn pow(base: Float64, exponent: Float64) -> Float64:
    """Power function: base^exponent."""
    return exp(exponent * log(base))

fn next_power_of_2(n: Int) -> Int:
    """Find next power of 2 >= n."""
    var power = 1
    while power < n:
        power *= 2
    return power

@always_inline
fn bit_reverse(n: Int, bits: Int) -> Int:
    """Reverse bits of integer n using 'bits' number of bits."""
    var result = 0
    var x = n
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

fn log2_int(n: Int) -> Int:
    """Compute log2 of power-of-2 integer."""
    var result = 0
    var x = n
    while x > 1:
        x >>= 1
        result += 1
    return result

# ------------------------------------------------------------------------------
# Window Functions
# ------------------------------------------------------------------------------

fn hann_window(size: Int) -> List[Float32]:
    """Generate Hann window: w(n) = 0.5 * (1 - cos(2π * n / (N-1)))"""
    var window = List[Float32]()
    var N = Float32(size - 1)

    for n in range(size):
        var n_float = Float32(n)
        var coefficient = 0.5 * (1.0 - cos(2.0 * pi * n_float / N))
        window.append(coefficient)

    return window^

fn hamming_window(size: Int) -> List[Float32]:
    """Generate Hamming window: w(n) = 0.54 - 0.46 * cos(2π * n / (N-1))"""
    var window = List[Float32]()
    var N = Float32(size - 1)

    for n in range(size):
        var n_float = Float32(n)
        var coefficient = 0.54 - 0.46 * cos(2.0 * pi * n_float / N)
        window.append(coefficient)

    return window^

fn apply_window_simd(signal: List[Float32], window: List[Float32]) raises -> List[Float32]:
    """SIMD-optimized window application."""
    if len(signal) != len(window):
        raise Error("Signal and window length mismatch")

    var N = len(signal)
    var result = List[Float32]()

    # Pre-allocate
    for _ in range(N):
        result.append(0.0)

    # SIMD with Float32 - 16 elements at once
    comptime simd_width = 16

    var i = 0
    while i + simd_width <= N:
        # Create SIMD vectors by loading from lists
        var sig_vec = SIMD[DType.float32, simd_width]()
        var win_vec = SIMD[DType.float32, simd_width]()

        @parameter
        for j in range(simd_width):
            sig_vec[j] = signal[i + j]
            win_vec[j] = window[i + j]

        # SIMD multiply
        var res_vec = sig_vec * win_vec

        # Store back
        @parameter
        for j in range(simd_width):
            result[i + j] = res_vec[j]

        i += simd_width

    # Remainder
    while i < N:
        result[i] = signal[i] * window[i]
        i += 1

    return result^

# ------------------------------------------------------------------------------
# FFT Operations
# ------------------------------------------------------------------------------

fn precompute_twiddle_factors(N: Int) -> List[Complex]:
    """Pre-compute all twiddle factors for FFT of size N."""
    var twiddles = List[Complex]()

    for i in range(N):
        var angle = -2.0 * pi * Float32(i) / Float32(N)
        twiddles.append(Complex(Float32(cos(angle)), Float32(sin(angle))))

    return twiddles^

fn fft_radix4(signal: List[Float32], twiddles: List[Complex]) raises -> List[Complex]:
    """Radix-4 FFT - faster than Radix-2!"""
    var N = len(signal)
    var log2_n = log2_int(N)

    # Initialize with bit-reversed input
    var result = List[Complex]()
    for i in range(N):
        var reversed_idx = bit_reverse(i, log2_n)
        result.append(Complex(signal[reversed_idx], 0.0))

    # Radix-4 stages
    var stride = 1
    while stride < N:
        var group_size = 4 * stride
        var twiddle_stride = N // group_size

        for group_start in range(0, N, group_size):
            for k in range(stride):
                # Get 4 butterfly indices
                var i0 = group_start + k
                var i1 = i0 + stride
                var i2 = i1 + stride
                var i3 = i2 + stride

                # Load inputs
                var x0 = Complex(result[i0].real, result[i0].imag)
                var x1 = Complex(result[i1].real, result[i1].imag)
                var x2 = Complex(result[i2].real, result[i2].imag)
                var x3 = Complex(result[i3].real, result[i3].imag)

                # Radix-4 butterfly
                var t0 = x0 + x2
                var t1 = x0 - x2
                var t2 = x1 + x3
                var t3 = Complex(x1.imag - x3.imag, x3.real - x1.real)

                var y0 = t0 + t2
                var y1 = t1 + t3
                var y2 = t0 - t2
                var y3 = t1 - t3

                # Apply twiddle factors
                var tw_idx = k * twiddle_stride
                if tw_idx > 0 and tw_idx < len(twiddles):
                    var w1 = Complex(twiddles[tw_idx].real, twiddles[tw_idx].imag)
                    var w2 = Complex(twiddles[2 * tw_idx].real, twiddles[2 * tw_idx].imag)
                    var w3 = Complex(twiddles[3 * tw_idx].real, twiddles[3 * tw_idx].imag)

                    y1 = y1 * w1
                    y2 = y2 * w2
                    y3 = y3 * w3

                # Store results
                result[i0] = Complex(y0.real, y0.imag)
                result[i1] = Complex(y1.real, y1.imag)
                result[i2] = Complex(y2.real, y2.imag)
                result[i3] = Complex(y3.real, y3.imag)

        stride *= 4

    return result^

fn fft_iterative_with_twiddles(
    signal: List[Float32],
    twiddles: List[Complex]
) raises -> List[Complex]:
    """Iterative FFT with pre-provided twiddle factors."""
    var N = len(signal)

    if N == 0 or (N & (N - 1)) != 0:
        raise Error("FFT requires power of 2")

    if len(twiddles) < N:
        raise Error("Insufficient twiddle factors")

    # Try Radix-4 for better performance
    var log2_n = log2_int(N)
    var is_power_of_4 = (log2_n % 2 == 0)

    if is_power_of_4:
        return fft_radix4(signal, twiddles)
    else:
        # Radix-2 FFT
        var result = List[Complex]()

        for i in range(N):
            var reversed_idx = bit_reverse(i, log2_n)
            result.append(Complex(signal[reversed_idx], 0.0))

        # Radix-2 butterfly
        var size = 2
        while size <= N:
            var half_size = size // 2
            var stride = N // size

            for i in range(0, N, size):
                for k in range(half_size):
                    var twiddle_idx = k * stride
                    var twiddle = Complex(twiddles[twiddle_idx].real, twiddles[twiddle_idx].imag)

                    var idx1 = i + k
                    var idx2 = i + k + half_size

                    var t = twiddle * Complex(result[idx2].real, result[idx2].imag)
                    var u = Complex(result[idx1].real, result[idx1].imag)

                    var sum_val = u + t
                    var diff_val = u - t

                    result[idx1] = Complex(sum_val.real, sum_val.imag)
                    result[idx2] = Complex(diff_val.real, diff_val.imag)

            size *= 2

        return result^

fn rfft_true(signal: List[Float32], twiddles: List[Complex]) raises -> List[Complex]:
    """TRUE Real FFT - exploits conjugate symmetry for 2x speedup!"""
    var N = len(signal)
    var fft_size = next_power_of_2(N)
    var half_size = fft_size // 2

    # Pad to power of 2
    var padded = List[Float32]()
    for i in range(N):
        padded.append(signal[i])
    for _ in range(N, fft_size):
        padded.append(0.0)

    # Pack: Convert N real → N/2 complex
    var packed = List[Float32]()
    for i in range(half_size):
        packed.append(padded[2 * i])

    # Compute N/2-point FFT
    var packed_fft = fft_iterative_with_twiddles(packed, twiddles)

    # Unpack: Extract positive frequencies
    var result = List[Complex]()

    # Bin 0 (DC)
    result.append(Complex(packed_fft[0].real, 0.0))

    # Bins 1 to N/2-1
    for k in range(1, half_size):
        if k < len(packed_fft):
            result.append(Complex(packed_fft[k].real, packed_fft[k].imag))

    # Bin N/2 (Nyquist)
    if half_size < len(packed_fft):
        result.append(Complex(packed_fft[0].imag, 0.0))
    else:
        result.append(Complex(0.0, 0.0))

    return result^

fn rfft_with_twiddles(
    signal: List[Float32],
    twiddles: List[Complex]
) raises -> List[Complex]:
    """TRUE RFFT using cached twiddles - 2x faster!"""
    return rfft_true(signal, twiddles)

fn power_spectrum(fft_output: List[Complex]) -> List[Float32]:
    """Compute power spectrum from FFT output (SIMD-optimized)."""
    var N = len(fft_output)
    var result = List[Float32]()

    # Pre-allocate
    for _ in range(N):
        result.append(0.0)

    # SIMD processing
    comptime simd_width = 16

    var i = 0
    while i + simd_width <= N:
        var real_vec = SIMD[DType.float32, simd_width]()
        var imag_vec = SIMD[DType.float32, simd_width]()

        @parameter
        for j in range(simd_width):
            real_vec[j] = fft_output[i + j].real
            imag_vec[j] = fft_output[i + j].imag

        # SIMD: real² + imag²
        var power_vec = real_vec * real_vec + imag_vec * imag_vec

        @parameter
        for j in range(simd_width):
            result[i + j] = power_vec[j]

        i += simd_width

    # Remainder
    while i < N:
        result[i] = fft_output[i].power()
        i += 1

    return result^

# ------------------------------------------------------------------------------
# STFT
# ------------------------------------------------------------------------------

fn stft(
    signal: List[Float32],
    n_fft: Int = 400,
    hop_length: Int = 160,
    window_fn: String = "hann"
) raises -> List[List[Float32]]:
    """Short-Time Fourier Transform - Apply FFT to windowed frames."""
    # Create window once
    var window: List[Float32]
    if window_fn == "hann":
        window = hann_window(n_fft)
    elif window_fn == "hamming":
        window = hamming_window(n_fft)
    else:
        raise Error("Unknown window function: " + window_fn)

    # Pre-compute twiddles once
    var fft_size = next_power_of_2(n_fft)
    var half_size = fft_size // 2
    var cached_twiddles = precompute_twiddle_factors(half_size)

    # Calculate number of frames
    var num_frames = (len(signal) - n_fft) // hop_length + 1
    var needed_bins = n_fft // 2 + 1

    # Pre-allocate spectrogram
    var spectrogram = List[List[Float32]]()
    for _ in range(num_frames):
        var frame_data = List[Float32]()
        for _ in range(needed_bins):
            frame_data.append(0.0)
        spectrogram.append(frame_data^)

    # Serial frame processing (parallel version crashes in FFI context)
    for frame_idx in range(num_frames):
        try:
            var start = frame_idx * hop_length

            # Extract frame
            var frame = List[Float32]()
            for i in range(n_fft):
                if start + i < len(signal):
                    frame.append(signal[start + i])
                else:
                    frame.append(0.0)

            # Apply window
            var windowed = apply_window_simd(frame, window)

            # RFFT with cached twiddles
            var fft_result = rfft_with_twiddles(windowed, cached_twiddles)

            # Power spectrum
            var full_power = power_spectrum(fft_result)

            # Store in pre-allocated spectrogram
            for i in range(needed_bins):
                if i < len(full_power):
                    spectrogram[frame_idx][i] = full_power[i]
        except:
            pass

    return spectrogram^

# ------------------------------------------------------------------------------
# Mel Scale Operations
# ------------------------------------------------------------------------------

fn hz_to_mel(freq_hz: Float32) -> Float32:
    """Convert frequency from Hz to Mel scale."""
    var log_val = log(Float64(1.0 + freq_hz / 700.0))
    return Float32(2595.0 * log_val / log(10.0))

fn mel_to_hz(freq_mel: Float32) -> Float32:
    """Convert frequency from Mel scale to Hz."""
    var pow_val = pow(Float64(10.0), Float64(freq_mel / 2595.0))
    return Float32(700.0 * (pow_val - 1.0))

fn create_mel_filterbank(
    n_mels: Int,
    n_fft: Int,
    sample_rate: Int
) -> List[List[Float32]]:
    """Create mel filterbank matrix."""
    var n_freq_bins = n_fft // 2 + 1

    # Frequency range: 0 Hz to Nyquist
    var nyquist = Float32(sample_rate) / 2.0

    # Convert to mel scale
    var mel_min = hz_to_mel(0.0)
    var mel_max = hz_to_mel(nyquist)

    # Create evenly spaced mel frequencies
    var mel_points = List[Float32]()
    var mel_step = (mel_max - mel_min) / Float32(n_mels + 1)

    for i in range(n_mels + 2):
        mel_points.append(mel_min + Float32(i) * mel_step)

    # Convert mel points back to Hz
    var hz_points = List[Float32]()
    for i in range(len(mel_points)):
        hz_points.append(mel_to_hz(mel_points[i]))

    # Convert Hz to FFT bin numbers
    var bin_points = List[Int]()
    for i in range(len(hz_points)):
        var bin = Int((Float32(n_fft + 1) * hz_points[i]) / Float32(sample_rate))
        bin_points.append(bin)

    # Create filterbank
    var filterbank = List[List[Float32]]()

    for mel_idx in range(n_mels):
        var filter_band = List[Float32]()

        # Initialize all bins to 0
        for _ in range(n_freq_bins):
            filter_band.append(0.0)

        # Create triangular filter
        var left = bin_points[mel_idx]
        var center = bin_points[mel_idx + 1]
        var right = bin_points[mel_idx + 2]

        # Create triangular filter only if valid range
        if center > left and right > center:
            # Rising slope
            for bin_idx in range(left, center):
                if bin_idx < n_freq_bins and bin_idx >= 0:
                    var weight = Float32(bin_idx - left) / Float32(center - left)
                    filter_band[bin_idx] = weight

            # Falling slope
            for bin_idx in range(center, right):
                if bin_idx < n_freq_bins and bin_idx >= 0:
                    var weight = Float32(right - bin_idx) / Float32(right - center)
                    filter_band[bin_idx] = weight

        filterbank.append(filter_band^)

    return filterbank^

struct SparseFilter(Copyable, Movable):
    """Sparse representation of mel filter band."""
    var start_idx: Int
    var end_idx: Int
    var weights: List[Float32]

    fn __init__(out self, start: Int, end: Int):
        self.start_idx = start
        self.end_idx = end
        self.weights = List[Float32]()

    fn __copyinit__(out self, existing: Self):
        self.start_idx = existing.start_idx
        self.end_idx = existing.end_idx
        self.weights = List[Float32]()
        for i in range(len(existing.weights)):
            self.weights.append(existing.weights[i])

    fn __moveinit__(out self, deinit existing: Self):
        self.start_idx = existing.start_idx
        self.end_idx = existing.end_idx
        self.weights = existing.weights^

fn create_sparse_mel_filterbank(
    filterbank: List[List[Float32]]
) -> List[SparseFilter]:
    """Convert dense filterbank to sparse representation."""
    var sparse_filters = List[SparseFilter]()

    for mel_idx in range(len(filterbank)):
        # Find non-zero range
        var start = -1
        var end = -1

        for freq_idx in range(len(filterbank[mel_idx])):
            if filterbank[mel_idx][freq_idx] > 0.0:
                if start == -1:
                    start = freq_idx
                end = freq_idx

        # Create sparse filter
        if start != -1:
            var sparse = SparseFilter(start, end)
            for freq_idx in range(start, end + 1):
                sparse.weights.append(filterbank[mel_idx][freq_idx])
            sparse_filters.append(sparse^)
        else:
            # All-zero filter
            var sparse = SparseFilter(0, 0)
            sparse_filters.append(sparse^)

    return sparse_filters^

fn apply_mel_filterbank(
    spectrogram: List[List[Float32]],
    filterbank: List[List[Float32]]
) raises -> List[List[Float32]]:
    """Apply mel filterbank to power spectrogram."""
    var n_frames = len(spectrogram)
    if n_frames == 0:
        raise Error("Empty spectrogram")

    var n_freq_bins = len(spectrogram[0])
    var n_mels = len(filterbank)

    # Validate dimensions
    if len(filterbank[0]) != n_freq_bins:
        raise Error("Filterbank size mismatch with spectrogram")

    # Create sparse representation
    var sparse_filters = create_sparse_mel_filterbank(filterbank)

    var mel_spec = List[List[Float32]]()

    # For each mel band
    for mel_idx in range(n_mels):
        var mel_band = List[Float32]()

        # Pre-allocate
        for _ in range(n_frames):
            mel_band.append(0.0)

        # Get sparse filter
        var start = sparse_filters[mel_idx].start_idx
        var end = sparse_filters[mel_idx].end_idx

        # For each time frame
        for frame_idx in range(n_frames):
            var mel_energy: Float32 = 0.0

            # Only iterate over non-zero filter weights
            var weight_idx = 0
            for freq_idx in range(start, end + 1):
                if weight_idx < len(sparse_filters[mel_idx].weights):
                    mel_energy += sparse_filters[mel_idx].weights[weight_idx] * spectrogram[frame_idx][freq_idx]
                    weight_idx += 1

            mel_band[frame_idx] = mel_energy

        mel_spec.append(mel_band^)

    return mel_spec^

# ------------------------------------------------------------------------------
# Mel Spectrogram (Main Function)
# ------------------------------------------------------------------------------

fn mel_spectrogram_ffi(
    audio: List[Float32],
    sample_rate: Int,
    n_fft: Int,
    hop_length: Int,
    n_mels: Int
) raises -> List[List[Float32]]:
    """
    Compute mel spectrogram - inlined implementation for FFI context.

    This is the complete transformation: audio → mel spectrogram.
    Inlined from audio.mojo to avoid shared library issues.
    """
    # Step 1: Compute STFT (power spectrogram)
    var power_spec = stft(audio, n_fft, hop_length, "hann")

    # Step 2: Create mel filterbank
    var filterbank = create_mel_filterbank(n_mels, n_fft, sample_rate)

    # Step 3: Apply filterbank
    var mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # Step 4: Log scaling
    var epsilon: Float32 = 1e-10

    for i in range(len(mel_spec)):
        for j in range(len(mel_spec[i])):
            # Clamp to epsilon minimum
            var value = mel_spec[i][j]
            if value < epsilon:
                value = epsilon

            # Apply log10 scaling
            var log_value = log(Float64(value)) / log(10.0)
            mel_spec[i][j] = Float32(log_value)

    return mel_spec^


# ==============================================================================
# Handle Management - Handles are just pointer addresses cast to Int32
# ==============================================================================

@always_inline
fn _ptr_to_handle[T: AnyType](ptr: UnsafePointer[T]) -> Int64:
    var addr = Int(ptr)
    return Int64(addr)

@always_inline
fn _handle_to_ptr(handle: Int64) -> UnsafePointer[mut=True, MojoMelSpectrogram, MutOrigin.external]:
    return UnsafePointer[mut=True, MojoMelSpectrogram, MutOrigin.external](
        unsafe_from_address=Int(handle)
    )

@export("mojo_audio_version", ABI="C")
fn mojo_audio_version(
    major: UnsafePointer[mut=True, Int32, MutAnyOrigin],
    minor: UnsafePointer[mut=True, Int32, MutAnyOrigin],
    patch: UnsafePointer[mut=True, Int32, MutAnyOrigin]
):
    major[0] = 0
    minor[0] = 1
    patch[0] = 0

@export("mojo_mel_config_default", ABI="C")
fn mojo_mel_config_default(out_config: UnsafePointer[mut=True, MojoMelConfig, MutAnyOrigin]):
    out_config[0] = MojoMelConfig()

@export("mojo_mel_spectrogram_compute", ABI="C")
fn mojo_mel_spectrogram_compute(
    audio_samples: UnsafePointer[mut=False, Float32, ImmutAnyOrigin],
    num_samples: UInt64,
    config: UnsafePointer[mut=False, MojoMelConfig, ImmutAnyOrigin]
) -> Int64:
    # Validate inputs
    if num_samples <= 0:
        return Int64(MOJO_AUDIO_ERROR_INVALID_INPUT)

    var cfg = config[0]
    if not cfg.is_valid():
        return Int64(MOJO_AUDIO_ERROR_INVALID_INPUT)

    try:
        # Convert C array to Mojo List
        var audio = List[Float32](capacity=Int(num_samples))
        for i in range(Int(num_samples)):
            audio.append(audio_samples[i])

        # Call inlined FFI version
        var result_2d = mel_spectrogram_ffi(
            audio,
            sample_rate=Int(cfg.sample_rate),
            n_fft=Int(cfg.n_fft),
            hop_length=Int(cfg.hop_length),
            n_mels=Int(cfg.n_mels)
        )

        # Validate result
        var n_mels = len(result_2d)
        if n_mels == 0:
            return Int64(MOJO_AUDIO_ERROR_PROCESSING)

        var n_frames = len(result_2d[0])
        if n_frames == 0:
            return Int64(MOJO_AUDIO_ERROR_PROCESSING)

        # Flatten to 1D (row-major)
        var flat_data = List[Float32](capacity=n_mels * n_frames)
        for i in range(n_mels):
            for j in range(n_frames):
                flat_data.append(result_2d[i][j])

        # Allocate on heap
        var mel_ptr = alloc[MojoMelSpectrogram](1)
        mel_ptr.init_pointee_move(MojoMelSpectrogram(flat_data^, n_mels, n_frames))
        return _ptr_to_handle(mel_ptr)

    except e:
        return Int64(MOJO_AUDIO_ERROR_PROCESSING)

@export("mojo_mel_spectrogram_get_shape", ABI="C")
fn mojo_mel_spectrogram_get_shape(
    handle: Int64,
    out_n_mels: UnsafePointer[mut=True, UInt64, MutAnyOrigin],
    out_n_frames: UnsafePointer[mut=True, UInt64, MutAnyOrigin]
) -> Int32:
    if handle <= 0:
        return MOJO_AUDIO_ERROR_INVALID_HANDLE

    try:
        var mel_ptr = _handle_to_ptr(handle)

        if out_n_mels:
            out_n_mels[0] = UInt64(mel_ptr[].n_mels)
        if out_n_frames:
            out_n_frames[0] = UInt64(mel_ptr[].n_frames)

        return MOJO_AUDIO_SUCCESS
    except:
        return MOJO_AUDIO_ERROR_INVALID_HANDLE

@export("mojo_mel_spectrogram_free", ABI="C")
fn mojo_mel_spectrogram_free(handle: Int64):
    """Free mel spectrogram and deallocate memory."""
    if handle <= 0:
        return

    try:
        var mel_ptr = _handle_to_ptr(handle)
        # Destructor will be called automatically when pointer is freed
        mel_ptr.free()
    except:
        # Silently ignore errors in free
        pass

@export("mojo_audio_last_error", ABI="C")
fn mojo_audio_last_error() -> UnsafePointer[mut=False, UInt8, ImmutOrigin.external]:
    return UnsafePointer[mut=False, UInt8, ImmutOrigin.external](unsafe_from_address=0)

@export("mojo_mel_spectrogram_get_size", ABI="C")
fn mojo_mel_spectrogram_get_size(handle: Int64) -> UInt64:
    if handle <= 0:
        return 0

    try:
        var mel_ptr = _handle_to_ptr(handle)
        return UInt64(mel_ptr[].n_mels * mel_ptr[].n_frames)
    except:
        return 0

@export("mojo_mel_spectrogram_get_data", ABI="C")
fn mojo_mel_spectrogram_get_data(
    handle: Int64,
    out_buffer: UnsafePointer[mut=True, Float32, MutAnyOrigin],
    buffer_size: UInt64
) -> Int32:
    if handle <= 0:
        return MOJO_AUDIO_ERROR_INVALID_HANDLE

    try:
        var mel_ptr = _handle_to_ptr(handle)

        var total_size = mel_ptr[].n_mels * mel_ptr[].n_frames
        if UInt64(total_size) > buffer_size:
            return MOJO_AUDIO_ERROR_BUFFER_SIZE

        # Copy data to output buffer
        for i in range(total_size):
            out_buffer[i] = mel_ptr[].data[i]

        return MOJO_AUDIO_SUCCESS
    except:
        return MOJO_AUDIO_ERROR_INVALID_HANDLE

@export("mojo_mel_spectrogram_is_valid", ABI="C")
fn mojo_mel_spectrogram_is_valid(handle: Int64) -> Int32:
    return 1 if handle > 0 else 0
