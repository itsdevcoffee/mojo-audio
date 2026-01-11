/**
 * C Example: Using mojo-audio FFI
 *
 * Demonstrates how to call mojo-audio from C code.
 *
 * Compile:
 *   gcc demo_c.c -I../../include -L../.. -lmojo_audio -lm -o demo_c
 *
 * Run:
 *   LD_LIBRARY_PATH=../.. ./demo_c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mojo_audio.h"

int main() {
    printf("=======================================================\n");
    printf("mojo-audio FFI Demo (C)\n");
    printf("=======================================================\n\n");

    /* Check library version */
    int32_t major, minor, patch;
    mojo_audio_version(&major, &minor, &patch);
    printf("Library version: %d.%d.%d\n\n", major, minor, patch);

    /* Create 30s of test audio (480k samples @ 16kHz) */
    size_t num_samples = 480000;
    float* audio = calloc(num_samples, sizeof(float));
    if (!audio) {
        fprintf(stderr, "Failed to allocate audio buffer\n");
        return 1;
    }

    /* Fill with 440Hz sine wave */
    printf("Generating 30s test audio (440Hz sine wave)...\n");
    for (size_t i = 0; i < num_samples; i++) {
        audio[i] = 0.5f * sinf(2.0f * M_PI * 440.0f * (float)i / 16000.0f);
    }
    printf("  Generated %zu samples\n\n", num_samples);

    /* Get default Whisper configuration */
    MojoMelConfig config = mojo_mel_config_default();
    printf("Configuration:\n");
    printf("  Sample rate: %d Hz\n", config.sample_rate);
    printf("  FFT size: %d\n", config.n_fft);
    printf("  Hop length: %d\n", config.hop_length);
    printf("  Mel bands: %d\n\n", config.n_mels);

    /* Validate configuration */
    MojoAudioStatus status = mojo_mel_config_validate(&config);
    if (status != MOJO_AUDIO_SUCCESS) {
        fprintf(stderr, "Config validation failed: %s\n", mojo_audio_last_error());
        free(audio);
        return 1;
    }

    /* Compute mel spectrogram */
    printf("Computing mel spectrogram...\n");
    MojoMelSpectrogram* mel = NULL;
    status = mojo_mel_spectrogram_compute(
        audio, num_samples, &config, &mel
    );

    if (status != MOJO_AUDIO_SUCCESS) {
        fprintf(stderr, "Error computing mel spectrogram: %s\n",
                mojo_audio_last_error());
        free(audio);
        return 1;
    }
    printf("  Computation successful!\n\n");

    /* Get result dimensions */
    size_t n_mels, n_frames;
    mojo_mel_spectrogram_get_shape(mel, &n_mels, &n_frames);
    printf("Result shape: (%zu, %zu)\n", n_mels, n_frames);

    size_t total_size = mojo_mel_spectrogram_get_size(mel);
    printf("Total elements: %zu\n\n", total_size);

    /* Allocate buffer and get data */
    float* mel_data = malloc(total_size * sizeof(float));
    if (!mel_data) {
        fprintf(stderr, "Failed to allocate mel data buffer\n");
        mojo_mel_spectrogram_free(mel);
        free(audio);
        return 1;
    }

    status = mojo_mel_spectrogram_get_data(mel, mel_data, total_size);
    if (status != MOJO_AUDIO_SUCCESS) {
        fprintf(stderr, "Error getting mel data: %s\n",
                mojo_audio_last_error());
        free(mel_data);
        mojo_mel_spectrogram_free(mel);
        free(audio);
        return 1;
    }

    /* Display first few values */
    printf("First 10 mel values:\n  ");
    for (int i = 0; i < 10 && i < (int)total_size; i++) {
        printf("%.4f ", mel_data[i]);
    }
    printf("\n\n");

    /* Compute statistics */
    float min_val = mel_data[0];
    float max_val = mel_data[0];
    double sum = 0.0;

    for (size_t i = 0; i < total_size; i++) {
        float val = mel_data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    float mean = (float)(sum / total_size);

    printf("Statistics:\n");
    printf("  Min: %.4f\n", min_val);
    printf("  Max: %.4f\n", max_val);
    printf("  Mean: %.4f\n\n", mean);

    /* Cleanup */
    free(mel_data);
    mojo_mel_spectrogram_free(mel);
    free(audio);

    printf("=======================================================\n");
    printf("Demo completed successfully!\n");
    printf("=======================================================\n");

    return 0;
}
