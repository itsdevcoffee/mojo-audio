"""
FFI type definitions for mojo-audio C interop.

Structs here must maintain exact C ABI compatibility.
See docs/context/01-11-2026-mojo-ffi-constraints.md for constraints.
"""


# ==============================================================================
# Normalization Constants (FFI-compatible Int32)
# ==============================================================================
# These match the enum values in mojo_audio.h

comptime MOJO_NORM_NONE: Int32 = 0      # Raw log mels, range [-10, 0]
comptime MOJO_NORM_WHISPER: Int32 = 1   # Whisper: clamp to max-8, (x+4)/4, range ~[-1, 1]
comptime MOJO_NORM_MINMAX: Int32 = 2    # Min-max scaling to [0, 1]
comptime MOJO_NORM_ZSCORE: Int32 = 3    # Z-score: (x - mean) / std, range ~[-3, 3]


@register_passable("trivial")
struct MojoMelConfig(Copyable, Movable):
    """Configuration matching C struct layout."""
    var sample_rate: Int32
    var n_fft: Int32
    var hop_length: Int32
    var n_mels: Int32
    var normalization: Int32  # MOJO_NORM_NONE, MOJO_NORM_WHISPER, etc.

    fn __init__(out self):
        """Create default Whisper-compatible configuration (raw log mels)."""
        self.sample_rate = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80
        self.normalization = MOJO_NORM_NONE  # Default: raw log mels (backwards compatible)

    fn is_valid(self) -> Bool:
        """Check if configuration parameters are valid."""
        return (
            self.sample_rate > 0 and
            self.n_fft > 0 and
            self.hop_length > 0 and
            self.n_mels > 0 and
            self.hop_length <= self.n_fft and
            self.normalization >= 0 and
            self.normalization <= 3
        )


struct MojoMelSpectrogram(Copyable, Movable):
    """Mel spectrogram result with metadata."""
    var data: List[Float32]
    var n_mels: Int
    var n_frames: Int

    fn __init__(out self, var data: List[Float32], n_mels: Int, n_frames: Int):
        """Create mel spectrogram with data and dimensions."""
        self.data = data^
        self.n_mels = n_mels
        self.n_frames = n_frames

    fn total_size(self) -> Int:
        """Get total number of elements (n_mels * n_frames)."""
        return self.n_mels * self.n_frames


# Error codes matching C enum
comptime MOJO_AUDIO_SUCCESS: Int32 = 0
comptime MOJO_AUDIO_ERROR_INVALID_INPUT: Int32 = -1
comptime MOJO_AUDIO_ERROR_ALLOCATION: Int32 = -2
comptime MOJO_AUDIO_ERROR_PROCESSING: Int32 = -3
comptime MOJO_AUDIO_ERROR_BUFFER_SIZE: Int32 = -4
comptime MOJO_AUDIO_ERROR_INVALID_HANDLE: Int32 = -5
