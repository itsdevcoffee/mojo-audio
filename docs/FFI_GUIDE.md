# FFI Integration Guide

**Using mojo-audio from C, Rust, Python, and other languages**

---

## 🎯 Overview

mojo-audio provides a **C-compatible FFI** (Foreign Function Interface) that allows you to call high-performance mel spectrogram computation from any language that supports C interop.

**Supported languages:**
- ✅ C/C++
- ✅ Rust
- ✅ Python (via ctypes/cffi)
- ✅ Go (via cgo)
- ✅ Any language with C FFI support

**Performance:**
- Same speed as native Mojo (12ms for 30s audio with -O3)
- Zero overhead - direct function calls
- Efficient memory management

---

## 🚀 Quick Start

### 1. Build the Shared Library

```bash
cd mojo-audio

# Build optimized version (recommended)
pixi run build-ffi-optimized

# Or build debug version
pixi run build-ffi

# This creates: libmojo_audio.so
```

### 2. Install Library and Headers

```bash
# Install to ~/.local (default)
pixi run install-ffi

# Or manually install to system directories (requires sudo)
sudo mkdir -p /usr/local/lib /usr/local/include
sudo cp libmojo_audio.so /usr/local/lib/
sudo cp include/mojo_audio.h /usr/local/include/
sudo ldconfig  # Update library cache
```

### 3. Use in Your Project

See language-specific examples below.

---

## 📖 API Reference

### Version Information

```c
void mojo_audio_version(int32_t* major, int32_t* minor, int32_t* patch);
```

Get library version at runtime.

### Configuration

```c
typedef struct {
    int32_t sample_rate;  // Audio sample rate (default: 16000 Hz)
    int32_t n_fft;        // FFT window size (default: 400)
    int32_t hop_length;   // Hop between frames (default: 160)
    int32_t n_mels;       // Number of mel bands (default: 80)
} MojoMelConfig;

// Get default Whisper-compatible config
MojoMelConfig mojo_mel_config_default(void);

// Validate configuration
MojoAudioStatus mojo_mel_config_validate(const MojoMelConfig* config);
```

### Mel Spectrogram Computation

```c
// Opaque handle
typedef struct MojoMelSpectrogram MojoMelSpectrogram;

// Compute mel spectrogram
MojoAudioStatus mojo_mel_spectrogram_compute(
    const float* audio_samples,
    size_t num_samples,
    const MojoMelConfig* config,
    MojoMelSpectrogram** out_result
);

// Get dimensions
void mojo_mel_spectrogram_get_shape(
    const MojoMelSpectrogram* mel,
    size_t* out_n_mels,
    size_t* out_n_frames
);

// Get total size
size_t mojo_mel_spectrogram_get_size(const MojoMelSpectrogram* mel);

// Copy data to buffer
MojoAudioStatus mojo_mel_spectrogram_get_data(
    const MojoMelSpectrogram* mel,
    float* out_buffer,
    size_t buffer_size
);

// Free result
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

// Get last error message
const char* mojo_audio_last_error(void);
```

---

## 💻 Language Examples

### C/C++

**File:** `demo.c`

```c
#include <stdio.h>
#include <stdlib.h>
#include "mojo_audio.h"

int main() {
    // Create audio samples
    float* audio = calloc(480000, sizeof(float));

    // Get default config
    MojoMelConfig config = mojo_mel_config_default();

    // Compute mel spectrogram
    MojoMelSpectrogram* mel = NULL;
    MojoAudioStatus status = mojo_mel_spectrogram_compute(
        audio, 480000, &config, &mel
    );

    if (status != MOJO_AUDIO_SUCCESS) {
        fprintf(stderr, "Error: %s\n", mojo_audio_last_error());
        free(audio);
        return 1;
    }

    // Get shape
    size_t n_mels, n_frames;
    mojo_mel_spectrogram_get_shape(mel, &n_mels, &n_frames);
    printf("Shape: (%zu, %zu)\n", n_mels, n_frames);

    // Get data
    size_t size = mojo_mel_spectrogram_get_size(mel);
    float* data = malloc(size * sizeof(float));
    mojo_mel_spectrogram_get_data(mel, data, size);

    // Cleanup
    free(data);
    mojo_mel_spectrogram_free(mel);
    free(audio);

    return 0;
}
```

**Compile:**
```bash
gcc demo.c -I/usr/local/include -L/usr/local/lib -lmojo_audio -lm -o demo
./demo
```

---

### Rust

**File:** `main.rs`

```rust
use std::ffi::CStr;

#[repr(C)]
struct MojoMelConfig {
    sample_rate: i32,
    n_fft: i32,
    hop_length: i32,
    n_mels: i32,
}

#[repr(C)]
struct MojoMelSpectrogram {
    _private: [u8; 0],
}

#[link(name = "mojo_audio")]
extern "C" {
    fn mojo_mel_config_default() -> MojoMelConfig;

    fn mojo_mel_spectrogram_compute(
        audio: *const f32,
        num_samples: usize,
        config: *const MojoMelConfig,
        out_result: *mut *mut MojoMelSpectrogram,
    ) -> i32;

    fn mojo_mel_spectrogram_get_shape(
        mel: *const MojoMelSpectrogram,
        out_n_mels: *mut usize,
        out_n_frames: *mut usize,
    );

    fn mojo_mel_spectrogram_get_data(
        mel: *const MojoMelSpectrogram,
        buffer: *mut f32,
        buffer_size: usize,
    ) -> i32;

    fn mojo_mel_spectrogram_free(mel: *mut MojoMelSpectrogram);

    fn mojo_audio_last_error() -> *const i8;
}

fn main() {
    unsafe {
        // Create audio
        let audio: Vec<f32> = vec![0.0; 480000];

        // Get config
        let config = mojo_mel_config_default();

        // Compute mel
        let mut mel: *mut MojoMelSpectrogram = std::ptr::null_mut();
        let status = mojo_mel_spectrogram_compute(
            audio.as_ptr(),
            audio.len(),
            &config,
            &mut mel,
        );

        if status != 0 {
            let err = CStr::from_ptr(mojo_audio_last_error());
            eprintln!("Error: {:?}", err);
            return;
        }

        // Get shape
        let mut n_mels: usize = 0;
        let mut n_frames: usize = 0;
        mojo_mel_spectrogram_get_shape(mel, &mut n_mels, &mut n_frames);
        println!("Shape: ({}, {})", n_mels, n_frames);

        // Get data
        let size = n_mels * n_frames;
        let mut data = vec![0.0f32; size];
        mojo_mel_spectrogram_get_data(mel, data.as_mut_ptr(), size);

        // Cleanup
        mojo_mel_spectrogram_free(mel);
    }
}
```

**Add to `Cargo.toml`:**
```toml
[build]
rustc-link-lib = ["mojo_audio"]
rustc-link-search = ["/usr/local/lib"]
```

**Or use `build.rs`:**
```rust
fn main() {
    println!("cargo:rustc-link-lib=mojo_audio");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}
```

**Build and run:**
```bash
cargo build
cargo run
```

---

### Python (ctypes)

**File:** `demo.py`

```python
from ctypes import *
import numpy as np

# Load library
lib = CDLL("libmojo_audio.so")  # or use /usr/local/lib/libmojo_audio.so

# Define structures
class MojoMelConfig(Structure):
    _fields_ = [
        ("sample_rate", c_int32),
        ("n_fft", c_int32),
        ("hop_length", c_int32),
        ("n_mels", c_int32),
    ]

class MojoMelSpectrogram(Structure):
    pass  # Opaque

# Define function signatures
lib.mojo_mel_config_default.restype = MojoMelConfig

lib.mojo_mel_spectrogram_compute.argtypes = [
    POINTER(c_float),
    c_size_t,
    POINTER(MojoMelConfig),
    POINTER(POINTER(MojoMelSpectrogram))
]
lib.mojo_mel_spectrogram_compute.restype = c_int32

lib.mojo_mel_spectrogram_get_shape.argtypes = [
    POINTER(MojoMelSpectrogram),
    POINTER(c_size_t),
    POINTER(c_size_t)
]

lib.mojo_mel_spectrogram_get_data.argtypes = [
    POINTER(MojoMelSpectrogram),
    POINTER(c_float),
    c_size_t
]
lib.mojo_mel_spectrogram_get_data.restype = c_int32

lib.mojo_mel_spectrogram_free.argtypes = [POINTER(MojoMelSpectrogram)]

# Use the library
audio = np.zeros(480000, dtype=np.float32)
config = lib.mojo_mel_config_default()

mel = POINTER(MojoMelSpectrogram)()
status = lib.mojo_mel_spectrogram_compute(
    audio.ctypes.data_as(POINTER(c_float)),
    len(audio),
    byref(config),
    byref(mel)
)

if status != 0:
    print("Error computing mel spectrogram")
else:
    n_mels = c_size_t()
    n_frames = c_size_t()
    lib.mojo_mel_spectrogram_get_shape(mel, byref(n_mels), byref(n_frames))

    print(f"Shape: ({n_mels.value}, {n_frames.value})")

    size = n_mels.value * n_frames.value
    data = np.zeros(size, dtype=np.float32)
    lib.mojo_mel_spectrogram_get_data(
        mel,
        data.ctypes.data_as(POINTER(c_float)),
        size
    )

    mel_2d = data.reshape(n_mels.value, n_frames.value)
    print(f"Mel spectrogram shape: {mel_2d.shape}")

    lib.mojo_mel_spectrogram_free(mel)
```

---

## 🔧 Integration Patterns

### Memory Management

**Caller-managed (recommended):**
```c
size_t size = mojo_mel_spectrogram_get_size(mel);
float* buffer = malloc(size * sizeof(float));
mojo_mel_spectrogram_get_data(mel, buffer, size);
// Use buffer...
free(buffer);
mojo_mel_spectrogram_free(mel);
```

**Zero-copy (advanced):**
```c
const float* data_ptr = mojo_mel_spectrogram_get_data_ptr(mel);
// Use data_ptr directly (read-only)
// WARNING: Invalid after mojo_mel_spectrogram_free()
mojo_mel_spectrogram_free(mel);
```

### Error Handling Pattern

```c
MojoAudioStatus status = mojo_mel_spectrogram_compute(...);
if (status != MOJO_AUDIO_SUCCESS) {
    const char* error = mojo_audio_last_error();
    fprintf(stderr, "Error (%d): %s\n", status, error);
    return status;
}
```

---

## 🎯 Real-World Usage: dev-voice Integration

See `examples/ffi/demo_rust.rs` for a complete Rust integration example.

**Key pattern for Whisper preprocessing:**

```rust
// Load your audio file
let audio_samples: Vec<f32> = load_audio();

// Configure for Whisper
let config = MojoMelConfig {
    sample_rate: 16000,
    n_fft: 400,
    hop_length: 160,
    n_mels: 80,
};

// Compute mel spectrogram
let mel = compute_mel_spectrogram(&audio_samples, &config)?;
let (n_mels, n_frames) = mel.shape(); // (80, ~3000)

// Get data for Whisper model
let mel_data = mel.to_vec()?;

// Feed to Candle Tensor
let tensor = Tensor::from_vec(mel_data, (n_mels, n_frames), &device)?;
```

---

## 📊 Performance Tips

1. **Use optimized build:** `pixi run build-ffi-optimized` (20-40% faster)
2. **Pre-allocate buffers:** Reuse mel data buffer across calls
3. **Batch processing:** Process multiple audio chunks efficiently
4. **Memory:**
   - 30s audio: ~480KB input, ~1MB output
   - Minimal overhead: ~50KB per call

---

## 🐛 Troubleshooting

### Library not found

```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or copy to system library directory
sudo cp libmojo_audio.so /usr/lib/
```

### Undefined symbols

Make sure you're linking against the library:
```bash
gcc ... -lmojo_audio -lm
```

### Version mismatch

Check header and library version match:
```c
int32_t major, minor, patch;
mojo_audio_version(&major, &minor, &patch);
// Compare with MOJO_AUDIO_VERSION_* from header
```

---

## 📚 Full Examples

See `examples/ffi/` directory for complete working examples:
- `demo_c.c` - C example with error handling
- `demo_rust.rs` - Rust safe wrapper
- `Makefile` - Build system

---

## 🤝 Contributing

If you add support for a new language, please contribute an example!

Languages we'd love to see:
- Go
- Julia
- Zig
- Swift
- JavaScript (Node.js N-API)

---

**Questions?** Open an issue on GitHub!
