/**
 * Rust Example: Using mojo-audio FFI
 *
 * Demonstrates safe Rust wrapper around mojo-audio C API.
 *
 * Compile:
 *   rustc -L ../.. -l mojo_audio demo_rust.rs -o demo_rust
 *
 * Run:
 *   LD_LIBRARY_PATH=../.. ./demo_rust
 */

use std::ffi::CStr;
use std::ptr;

// ============================================================================
// FFI Bindings
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MojoMelConfig {
    pub sample_rate: i32,
    pub n_fft: i32,
    pub hop_length: i32,
    pub n_mels: i32,
}

#[repr(C)]
pub struct MojoMelSpectrogram {
    _private: [u8; 0], // Opaque type
}

#[link(name = "mojo_audio")]
extern "C" {
    fn mojo_audio_version(major: *mut i32, minor: *mut i32, patch: *mut i32);

    fn mojo_audio_last_error() -> *const i8;

    fn mojo_mel_config_default() -> MojoMelConfig;

    fn mojo_mel_config_validate(config: *const MojoMelConfig) -> i32;

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

    fn mojo_mel_spectrogram_get_size(mel: *const MojoMelSpectrogram) -> usize;

    fn mojo_mel_spectrogram_get_data(
        mel: *const MojoMelSpectrogram,
        buffer: *mut f32,
        buffer_size: usize,
    ) -> i32;

    fn mojo_mel_spectrogram_free(mel: *mut MojoMelSpectrogram);
}

// ============================================================================
// Safe Rust Wrapper
// ============================================================================

pub struct MelSpectrogram {
    handle: *mut MojoMelSpectrogram,
}

impl MelSpectrogram {
    /// Compute mel spectrogram from audio samples
    pub fn compute(audio: &[f32], config: &MojoMelConfig) -> Result<Self, String> {
        unsafe {
            let mut handle: *mut MojoMelSpectrogram = ptr::null_mut();

            let status = mojo_mel_spectrogram_compute(
                audio.as_ptr(),
                audio.len(),
                config,
                &mut handle,
            );

            if status != 0 {
                let err_ptr = mojo_audio_last_error();
                let err_msg = CStr::from_ptr(err_ptr)
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("Mojo audio error: {}", err_msg));
            }

            Ok(MelSpectrogram { handle })
        }
    }

    /// Get dimensions (n_mels, n_frames)
    pub fn shape(&self) -> (usize, usize) {
        unsafe {
            let mut n_mels: usize = 0;
            let mut n_frames: usize = 0;
            mojo_mel_spectrogram_get_shape(self.handle, &mut n_mels, &mut n_frames);
            (n_mels, n_frames)
        }
    }

    /// Get total size (n_mels * n_frames)
    pub fn size(&self) -> usize {
        unsafe { mojo_mel_spectrogram_get_size(self.handle) }
    }

    /// Copy data to Vec
    pub fn to_vec(&self) -> Result<Vec<f32>, String> {
        let size = self.size();
        let mut data = vec![0.0f32; size];

        unsafe {
            let status = mojo_mel_spectrogram_get_data(
                self.handle,
                data.as_mut_ptr(),
                size,
            );

            if status != 0 {
                return Err("Failed to copy mel data".to_string());
            }
        }

        Ok(data)
    }
}

impl Drop for MelSpectrogram {
    fn drop(&mut self) {
        unsafe {
            mojo_mel_spectrogram_free(self.handle);
        }
    }
}

// ============================================================================
// Demo Application
// ============================================================================

fn main() {
    println!("=======================================================");
    println!("mojo-audio FFI Demo (Rust)");
    println!("=======================================================\n");

    // Check library version
    unsafe {
        let mut major = 0i32;
        let mut minor = 0i32;
        let mut patch = 0i32;
        mojo_audio_version(&mut major, &mut minor, &mut patch);
        println!("Library version: {}.{}.{}\n", major, minor, patch);
    }

    // Create 30s of test audio (480k samples @ 16kHz)
    let num_samples = 480_000;
    println!("Generating 30s test audio (440Hz sine wave)...");
    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin()
        })
        .collect();
    println!("  Generated {} samples\n", audio.len());

    // Get default Whisper configuration
    let config = unsafe { mojo_mel_config_default() };
    println!("Configuration:");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  FFT size: {}", config.n_fft);
    println!("  Hop length: {}", config.hop_length);
    println!("  Mel bands: {}\n", config.n_mels);

    // Validate configuration
    let status = unsafe { mojo_mel_config_validate(&config) };
    if status != 0 {
        eprintln!("Config validation failed!");
        return;
    }

    // Compute mel spectrogram
    println!("Computing mel spectrogram...");
    let mel = match MelSpectrogram::compute(&audio, &config) {
        Ok(mel) => {
            println!("  Computation successful!\n");
            mel
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    // Get result dimensions
    let (n_mels, n_frames) = mel.shape();
    println!("Result shape: ({}, {})", n_mels, n_frames);
    println!("Total elements: {}\n", mel.size());

    // Get data
    let mel_data = match mel.to_vec() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error getting data: {}", e);
            return;
        }
    };

    // Display first few values
    print!("First 10 mel values:\n  ");
    for (i, &val) in mel_data.iter().take(10).enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{:.4}", val);
    }
    println!("\n");

    // Compute statistics
    let min_val = mel_data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = mel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = mel_data.iter().sum();
    let mean = sum / mel_data.len() as f32;

    println!("Statistics:");
    println!("  Min: {:.4}", min_val);
    println!("  Max: {:.4}", max_val);
    println!("  Mean: {:.4}\n", mean);

    println!("=======================================================");
    println!("Demo completed successfully!");
    println!("=======================================================");
}
