/**
 * mojo-audio FFI - C API for high-performance audio DSP
 *
 * Handle-based API design for maximum compatibility and safety.
 */

#ifndef MOJO_AUDIO_H
#define MOJO_AUDIO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version Information
 * ========================================================================== */

#define MOJO_AUDIO_VERSION_MAJOR 0
#define MOJO_AUDIO_VERSION_MINOR 1
#define MOJO_AUDIO_VERSION_PATCH 0

void mojo_audio_version(int32_t* major, int32_t* minor, int32_t* patch);

/* ============================================================================
 * Error Codes
 * ========================================================================== */

typedef enum {
    MOJO_AUDIO_SUCCESS = 0,
    MOJO_AUDIO_ERROR_INVALID_INPUT = -1,
    MOJO_AUDIO_ERROR_ALLOCATION = -2,
    MOJO_AUDIO_ERROR_PROCESSING = -3,
    MOJO_AUDIO_ERROR_BUFFER_SIZE = -4,
    MOJO_AUDIO_ERROR_INVALID_HANDLE = -5,
} MojoAudioStatus;

const char* mojo_audio_last_error(void);

/* ============================================================================
 * Configuration
 * ========================================================================== */

typedef struct {
    int32_t sample_rate;  /* Default: 16000 Hz */
    int32_t n_fft;        /* Default: 400 */
    int32_t hop_length;   /* Default: 160 */
    int32_t n_mels;       /* Default: 80 */
} MojoMelConfig;

void mojo_mel_config_default(MojoMelConfig* out_config);

/* ============================================================================
 * Mel Spectrogram - Handle-Based API
 * ========================================================================== */

/**
 * Handle to mel spectrogram result.
 *
 * Positive values are valid handles.
 * Zero is never returned.
 * Negative values are error codes.
 */
typedef int64_t MojoMelHandle;

/**
 * Compute mel spectrogram from audio samples.
 *
 * Returns:
 *   Positive handle on success (use with get_shape, get_data, free)
 *   Negative error code on failure (see MojoAudioStatus enum)
 *
 * Example:
 *   MojoMelHandle handle = mojo_mel_spectrogram_compute(audio, 480000, &config);
 *   if (handle > 0) {
 *       // Success - use handle
 *       mojo_mel_spectrogram_free(handle);
 *   } else {
 *       // Error
 *       fprintf(stderr, "Error: %s\n", mojo_audio_last_error());
 *   }
 *
 * Thread safety: This function is thread-safe. Multiple threads can
 * compute mel spectrograms concurrently.
 */
MojoMelHandle mojo_mel_spectrogram_compute(
    const float* audio_samples,
    size_t num_samples,
    const MojoMelConfig* config
);

/**
 * Get dimensions of mel spectrogram.
 *
 * Returns:
 *   MOJO_AUDIO_SUCCESS (0) on success
 *   MOJO_AUDIO_ERROR_INVALID_HANDLE if handle is invalid
 */
MojoAudioStatus mojo_mel_spectrogram_get_shape(
    MojoMelHandle handle,
    size_t* out_n_mels,
    size_t* out_n_frames
);

/**
 * Get total number of elements.
 *
 * Returns:
 *   Number of elements (n_mels * n_frames) on success
 *   0 if handle is invalid
 */
size_t mojo_mel_spectrogram_get_size(MojoMelHandle handle);

/**
 * Copy mel spectrogram data to caller-provided buffer.
 *
 * Data is in row-major order: buffer[i * n_frames + j] = mel[i][j]
 *
 * Returns:
 *   MOJO_AUDIO_SUCCESS if buffer_size >= get_size(handle)
 *   MOJO_AUDIO_ERROR_BUFFER_SIZE if buffer too small
 *   MOJO_AUDIO_ERROR_INVALID_HANDLE if handle is invalid
 */
MojoAudioStatus mojo_mel_spectrogram_get_data(
    MojoMelHandle handle,
    float* out_buffer,
    size_t buffer_size
);

/**
 * Free mel spectrogram result.
 *
 * After calling this, the handle becomes invalid and must not be used.
 * Calling with an already-freed or invalid handle is a no-op (safe).
 *
 * Thread safety: This function is thread-safe.
 */
void mojo_mel_spectrogram_free(MojoMelHandle handle);

/**
 * Check if a handle is valid.
 *
 * Returns:
 *   1 if handle is valid
 *   0 if handle is invalid or already freed
 *
 * Note: This is primarily for debugging. In production code,
 * you should track handle validity yourself.
 */
int mojo_mel_spectrogram_is_valid(MojoMelHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* MOJO_AUDIO_H */
