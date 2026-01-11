/**
 * mojo-audio FFI Demo (C)
 *
 * Demonstrates the handle-based C API for computing mel spectrograms.
 *
 * Build:
 *   gcc demo_c.c -I../../include -L../.. -lmojo_audio -lm -o demo_c
 *
 * Run:
 *   LD_LIBRARY_PATH=../.. ./demo_c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mojo_audio.h"

#define SAMPLE_RATE 16000
#define DURATION_SEC 30
#define FREQUENCY_HZ 440.0f

int main(void) {
    printf("mojo-audio FFI Demo (C)\n");
    printf("=======================\n\n");

    // Library version
    int32_t major, minor, patch;
    mojo_audio_version(&major, &minor, &patch);
    printf("Library version: %d.%d.%d\n\n", major, minor, patch);

    // Generate test audio: 30s sine wave at 440Hz
    size_t num_samples = SAMPLE_RATE * DURATION_SEC;
    float* audio = malloc(num_samples * sizeof(float));
    if (!audio) {
        fprintf(stderr, "Failed to allocate audio buffer\n");
        return 1;
    }

    printf("Generating %ds test audio (%dHz sine wave)...\n",
           DURATION_SEC, (int)FREQUENCY_HZ);
    for (size_t i = 0; i < num_samples; i++) {
        float t = (float)i / SAMPLE_RATE;
        audio[i] = 0.5f * sinf(2.0f * (float)M_PI * FREQUENCY_HZ * t);
    }
    printf("  %zu samples generated\n\n", num_samples);

    // Get default configuration (Whisper-compatible)
    MojoMelConfig config;
    mojo_mel_config_default(&config);
    printf("Configuration:\n");
    printf("  Sample rate: %d Hz\n", config.sample_rate);
    printf("  FFT size:    %d\n", config.n_fft);
    printf("  Hop length:  %d\n", config.hop_length);
    printf("  Mel bands:   %d\n\n", config.n_mels);

    // Compute mel spectrogram - returns handle (positive) or error (negative)
    printf("Computing mel spectrogram...\n");
    MojoMelHandle handle = mojo_mel_spectrogram_compute(audio, num_samples, &config);

    if (handle <= 0) {
        fprintf(stderr, "Error: %s\n", mojo_audio_last_error());
        free(audio);
        return 1;
    }
    printf("  Success! Handle: %ld\n\n", (long)handle);

    // Get dimensions
    size_t n_mels, n_frames;
    mojo_mel_spectrogram_get_shape(handle, &n_mels, &n_frames);
    size_t total_size = mojo_mel_spectrogram_get_size(handle);

    printf("Result:\n");
    printf("  Shape: (%zu, %zu)\n", n_mels, n_frames);
    printf("  Total: %zu elements\n\n", total_size);

    // Copy data to local buffer
    float* mel_data = malloc(total_size * sizeof(float));
    if (!mel_data) {
        fprintf(stderr, "Failed to allocate output buffer\n");
        mojo_mel_spectrogram_free(handle);
        free(audio);
        return 1;
    }

    MojoAudioStatus status = mojo_mel_spectrogram_get_data(handle, mel_data, total_size);
    if (status != MOJO_AUDIO_SUCCESS) {
        fprintf(stderr, "Error copying data: %s\n", mojo_audio_last_error());
        free(mel_data);
        mojo_mel_spectrogram_free(handle);
        free(audio);
        return 1;
    }

    // Show sample values
    printf("First 8 values: ");
    for (int i = 0; i < 8 && i < (int)total_size; i++) {
        printf("%.3f ", mel_data[i]);
    }
    printf("\n\n");

    // Compute statistics
    float min_val = mel_data[0], max_val = mel_data[0];
    double sum = 0.0;
    for (size_t i = 0; i < total_size; i++) {
        if (mel_data[i] < min_val) min_val = mel_data[i];
        if (mel_data[i] > max_val) max_val = mel_data[i];
        sum += mel_data[i];
    }

    printf("Statistics:\n");
    printf("  Min:  %.4f\n", min_val);
    printf("  Max:  %.4f\n", max_val);
    printf("  Mean: %.4f\n\n", (float)(sum / total_size));

    // Cleanup
    free(mel_data);
    mojo_mel_spectrogram_free(handle);
    free(audio);

    printf("Demo complete.\n");
    return 0;
}
