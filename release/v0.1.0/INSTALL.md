# mojo-audio v0.1.0 - Installation Guide

Pre-built FFI binaries for using mojo-audio from C, Rust, Python, or any language with C interop.

---

## 📦 Release Contents

```
mojo-audio-v0.1.0-linux-x86_64/
├── libmojo_audio.so              # Shared library (26KB, optimized)
├── mojo_audio.h                  # C header file
└── INSTALL.md                    # This file

mojo-audio-ffi-examples.tar.gz    # FFI usage examples (C, Rust)
```

---

## 🚀 Quick Install (Linux x86_64)

### System-wide Installation (Recommended)

```bash
# Install library and header
sudo cp libmojo_audio.so /usr/local/lib/
sudo cp mojo_audio.h /usr/local/include/

# Update library cache
sudo ldconfig

# Verify installation
ldconfig -p | grep mojo_audio
```

### User Installation

```bash
# Install to user directory
mkdir -p ~/.local/lib ~/.local/include
cp libmojo_audio.so ~/.local/lib/
cp mojo_audio.h ~/.local/include/

# Add to library path (add to ~/.bashrc for persistence)
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

---

## 🧪 Test Installation

### C Example

```bash
# Extract examples
tar xzf mojo-audio-ffi-examples.tar.gz
cd ffi-examples/

# Compile and run C example
gcc demo_c.c -I/usr/local/include -L/usr/local/lib -lmojo_audio -lm -o demo_c
./demo_c
```

**Expected output:**
```
mojo-audio v0.1.0
Computing mel spectrogram...
Shape: (80, 3001)
Success!
```

### Rust Example

```bash
# Create Rust project
cargo new --bin mojo-audio-test
cd mojo-audio-test

# Copy demo code
cp ../demo_rust.rs src/main.rs

# Create build.rs
cat > build.rs << 'EOF'
fn main() {
    println!("cargo:rustc-link-lib=mojo_audio");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
EOF

# Build and run
cargo build
cargo run
```

### Python Example

```python
from ctypes import *
import numpy as np

# Load library
lib = CDLL("/usr/local/lib/libmojo_audio.so")

# Define config structure
class MojoMelConfig(Structure):
    _fields_ = [
        ("sample_rate", c_int32),
        ("n_fft", c_int32),
        ("hop_length", c_int32),
        ("n_mels", c_int32),
    ]

# Get default config
lib.mojo_mel_config_default.restype = MojoMelConfig
config = lib.mojo_mel_config_default()

print(f"Config: {config.sample_rate}Hz, FFT={config.n_fft}, mels={config.n_mels}")
```

---

## 📊 Performance

This build is optimized with `-O3` and delivers:

- **1 second audio:** 1.1ms (3.6x faster than librosa)
- **10 second audio:** 7.5ms (2.0x faster than librosa)
- **30 second audio:** 27.4ms (1.1x faster than librosa)

Zero overhead compared to native Mojo - direct function calls!

---

## 🔧 API Overview

### Version Information

```c
void mojo_audio_version(int32_t* major, int32_t* minor, int32_t* patch);
```

### Configuration

```c
typedef struct {
    int32_t sample_rate;  // Default: 16000 Hz
    int32_t n_fft;        // Default: 400
    int32_t hop_length;   // Default: 160
    int32_t n_mels;       // Default: 80
    int32_t normalization; // 0=none, 1=whisper, 2=minmax, 3=zscore
} MojoMelConfig;

MojoMelConfig mojo_mel_config_default(void);
```

### Mel Spectrogram Computation

```c
typedef struct MojoMelSpectrogram MojoMelSpectrogram;

MojoAudioStatus mojo_mel_spectrogram_compute(
    const float* audio_samples,
    size_t num_samples,
    const MojoMelConfig* config,
    MojoMelSpectrogram** out_result
);

void mojo_mel_spectrogram_get_shape(
    const MojoMelSpectrogram* mel,
    size_t* out_n_mels,
    size_t* out_n_frames
);

MojoAudioStatus mojo_mel_spectrogram_get_data(
    const MojoMelSpectrogram* mel,
    float* out_buffer,
    size_t buffer_size
);

void mojo_mel_spectrogram_free(MojoMelSpectrogram* mel);
```

### Error Handling

```c
typedef enum {
    MOJO_AUDIO_SUCCESS = 0,
    MOJO_AUDIO_ERROR_INVALID_INPUT = -1,
    MOJO_AUDIO_ERROR_ALLOCATION = -2,
    MOJO_AUDIO_ERROR_PROCESSING = -3,
    MOJO_AUDIO_ERROR_BUFFER_SIZE = -4,
} MojoAudioStatus;

const char* mojo_audio_last_error(void);
```

---

## 📚 Full Documentation

For complete API documentation, usage examples, and language-specific guides:

**[FFI Integration Guide](https://github.com/itsdevcoffee/mojo-audio/blob/main/docs/guides/01-10-2026-ffi-guide.md)**

---

## 🐛 Troubleshooting

### Library not found

```bash
# Check if installed
ls /usr/local/lib/libmojo_audio.so

# Check library cache
ldconfig -p | grep mojo_audio

# If not in cache
sudo ldconfig

# Or add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Undefined symbols during linking

Make sure to link against the library:
```bash
gcc ... -lmojo_audio -lm
```

### Version mismatch

Check header and library versions match:
```c
int32_t major, minor, patch;
mojo_audio_version(&major, &minor, &patch);
printf("Library version: %d.%d.%d\n", major, minor, patch);
// Should output: Library version: 0.1.0
```

---

## 🏗️ Build Information

- **Version:** 0.1.0
- **Platform:** Linux x86_64
- **Mojo Version:** 0.26.1
- **Optimization:** `-O3` (maximum performance)
- **Build Date:** 2026-01-22
- **Size:** 26KB (stripped shared library)

---

## 🎯 Whisper Compatibility

This library is fully compatible with OpenAI Whisper models:

- ✅ Sample rate: 16kHz
- ✅ FFT size: 400
- ✅ Hop length: 160 (10ms frames)
- ✅ Mel bands: 80 (Whisper v2) or 128 (Whisper v3)
- ✅ Normalization: Whisper-compatible mode

---

## 📝 License

MIT License - See LICENSE file in the repository

---

## 🔗 Links

- **Repository:** https://github.com/itsdevcoffee/mojo-audio
- **Issues:** https://github.com/itsdevcoffee/mojo-audio/issues
- **FFI Guide:** https://github.com/itsdevcoffee/mojo-audio/blob/main/docs/guides/01-10-2026-ffi-guide.md

---

**Questions?** Open an issue on GitHub or check the documentation!
