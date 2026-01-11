//! mojo-audio FFI Demo (Rust)
//!
//! Demonstrates safe Rust bindings for the handle-based C API.
//!
//! Build:
//!   rustc -L ../.. -l mojo_audio demo_rust.rs -o demo_rust
//!
//! Run:
//!   LD_LIBRARY_PATH=../.. ./demo_rust

use std::ffi::CStr;

const SAMPLE_RATE: u32 = 16000;
const DURATION_SEC: u32 = 30;
const FREQUENCY_HZ: f32 = 440.0;

// =============================================================================
// FFI Bindings (matches mojo_audio.h)
// =============================================================================

type MojoMelHandle = i64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MojoMelConfig {
    sample_rate: i32,
    n_fft: i32,
    hop_length: i32,
    n_mels: i32,
}

#[link(name = "mojo_audio")]
extern "C" {
    fn mojo_audio_version(major: *mut i32, minor: *mut i32, patch: *mut i32);
    fn mojo_audio_last_error() -> *const i8;

    fn mojo_mel_config_default(out_config: *mut MojoMelConfig);

    fn mojo_mel_spectrogram_compute(
        audio: *const f32,
        num_samples: usize,
        config: *const MojoMelConfig,
    ) -> MojoMelHandle;

    fn mojo_mel_spectrogram_get_shape(
        handle: MojoMelHandle,
        out_n_mels: *mut usize,
        out_n_frames: *mut usize,
    ) -> i32;

    fn mojo_mel_spectrogram_get_size(handle: MojoMelHandle) -> usize;

    fn mojo_mel_spectrogram_get_data(
        handle: MojoMelHandle,
        buffer: *mut f32,
        buffer_size: usize,
    ) -> i32;

    fn mojo_mel_spectrogram_free(handle: MojoMelHandle);
}

// =============================================================================
// Safe Wrapper
// =============================================================================

struct MelSpectrogram {
    handle: MojoMelHandle,
}

impl MelSpectrogram {
    fn compute(audio: &[f32], config: &MojoMelConfig) -> Result<Self, String> {
        let handle = unsafe { mojo_mel_spectrogram_compute(audio.as_ptr(), audio.len(), config) };

        if handle <= 0 {
            let msg = unsafe {
                CStr::from_ptr(mojo_audio_last_error())
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(msg);
        }

        Ok(Self { handle })
    }

    fn shape(&self) -> (usize, usize) {
        let (mut n_mels, mut n_frames) = (0, 0);
        unsafe { mojo_mel_spectrogram_get_shape(self.handle, &mut n_mels, &mut n_frames) };
        (n_mels, n_frames)
    }

    fn size(&self) -> usize {
        unsafe { mojo_mel_spectrogram_get_size(self.handle) }
    }

    fn to_vec(&self) -> Result<Vec<f32>, &'static str> {
        let size = self.size();
        let mut data = vec![0.0f32; size];

        let status = unsafe { mojo_mel_spectrogram_get_data(self.handle, data.as_mut_ptr(), size) };

        if status != 0 {
            return Err("Failed to copy mel data");
        }

        Ok(data)
    }
}

impl Drop for MelSpectrogram {
    fn drop(&mut self) {
        unsafe { mojo_mel_spectrogram_free(self.handle) };
    }
}

// =============================================================================
// Demo
// =============================================================================

fn main() {
    println!("mojo-audio FFI Demo (Rust)");
    println!("==========================\n");

    // Library version
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    unsafe { mojo_audio_version(&mut major, &mut minor, &mut patch) };
    println!("Library version: {major}.{minor}.{patch}\n");

    // Generate test audio: 30s sine wave at 440Hz
    let num_samples = (SAMPLE_RATE * DURATION_SEC) as usize;
    println!("Generating {DURATION_SEC}s test audio ({FREQUENCY_HZ}Hz sine wave)...");

    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.5 * (2.0 * std::f32::consts::PI * FREQUENCY_HZ * t).sin()
        })
        .collect();
    println!("  {} samples generated\n", audio.len());

    // Get default configuration (Whisper-compatible)
    let mut config = MojoMelConfig {
        sample_rate: 0,
        n_fft: 0,
        hop_length: 0,
        n_mels: 0,
    };
    unsafe { mojo_mel_config_default(&mut config) };

    println!("Configuration:");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  FFT size:    {}", config.n_fft);
    println!("  Hop length:  {}", config.hop_length);
    println!("  Mel bands:   {}\n", config.n_mels);

    // Compute mel spectrogram
    println!("Computing mel spectrogram...");
    let mel = match MelSpectrogram::compute(&audio, &config) {
        Ok(m) => {
            println!("  Success!\n");
            m
        }
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    // Get dimensions
    let (n_mels, n_frames) = mel.shape();
    println!("Result:");
    println!("  Shape: ({n_mels}, {n_frames})");
    println!("  Total: {} elements\n", mel.size());

    // Copy data
    let data = match mel.to_vec() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    // Show sample values
    print!("First 8 values: ");
    for val in data.iter().take(8) {
        print!("{val:.3} ");
    }
    println!("\n");

    // Compute statistics
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;

    println!("Statistics:");
    println!("  Min:  {min_val:.4}");
    println!("  Max:  {max_val:.4}");
    println!("  Mean: {mean:.4}\n");

    println!("Demo complete.");
}
