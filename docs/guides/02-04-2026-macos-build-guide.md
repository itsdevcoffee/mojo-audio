# macOS Build Guide

**Last Updated:** February 4, 2026

This guide covers building mojo-audio on macOS (both Apple Silicon and Intel).

---

## Quick Start - Build on Your MacBook

### Prerequisites

- macOS Ventura (13) or later
- Apple Silicon (M1/M2/M3) or Intel processor
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew: https://brew.sh
- Pixi: https://pixi.sh

### Build Steps

```bash
# Clone the repository
git clone https://github.com/itsdevcoffee/mojo-audio.git
cd mojo-audio

# Install dependencies (this will now work on macOS!)
pixi install

# Run tests to verify everything works
pixi run test

# Build the FFI library (creates libmojo_audio.dylib)
pixi run build-ffi-optimized

# Verify the build
ls -lh libmojo_audio.dylib
```

### Platform-Specific Notes

**Apple Silicon (M1/M2/M3):**
- Uses `osx-arm64` platform
- Native performance, no emulation needed

**Intel Macs:**
- Uses `osx-64` platform
- If a package isn't available for your platform, pixi will automatically fall back

**Shared Library Extension:**
- macOS uses `.dylib` instead of `.so`
- The platform-specific tasks in `pixi.toml` handle this automatically

---

## Automated Builds with GitHub Actions

The repository now includes GitHub Actions workflows that automatically build releases for:
- ✅ Linux x86_64
- ✅ macOS Apple Silicon (arm64)
- ✅ macOS Intel (x86_64)

### Triggering a Release Build

When you create a new version tag, GitHub Actions will automatically build all platforms:

```bash
# Create and push a new version tag
git tag v0.2.0
git push origin v0.2.0
```

This triggers the workflow which:
1. Builds optimized FFI libraries for all platforms
2. Packages them with headers and installation instructions
3. Creates a GitHub Release with all artifacts attached

### Manual Testing

You can also trigger the workflow manually from GitHub:
1. Go to Actions tab in GitHub
2. Select "Build and Release" workflow
3. Click "Run workflow"

---

## Local Installation

After building, install the library system-wide:

```bash
# Install the dylib and header
sudo cp libmojo_audio.dylib /usr/local/lib/
sudo cp include/mojo_audio.h /usr/local/include/

# Verify installation
ls -l /usr/local/lib/libmojo_audio.dylib
ls -l /usr/local/include/mojo_audio.h
```

---

## Using the Library

### C Example

```c
#include <mojo_audio.h>

int main() {
    float audio[16000];  // 1 second at 16kHz
    // ... populate audio data ...

    float* mel_spec;
    int rows, cols;

    int result = compute_mel_spectrogram(
        audio, 16000,
        80,      // n_mels
        400,     // n_fft
        160,     // hop_length
        &mel_spec, &rows, &cols
    );

    // Use mel_spec...

    free_mel_spectrogram(mel_spec);
    return 0;
}
```

Compile on macOS:
```bash
gcc -o demo demo.c -lmojo_audio
./demo
```

---

## Troubleshooting

### "dylib not found" when running

Add to your shell profile (~/.zshrc or ~/.bash_profile):
```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### Pixi fails to resolve dependencies

Try specifying the environment explicitly:
```bash
pixi install --environment default
```

### Build errors with Mojo

Ensure you have the latest Mojo version:
```bash
pixi update mojo max
```

---

## What Changed

### pixi.toml Updates

1. **Platform support expanded:**
   ```toml
   platforms = ["linux-64", "osx-arm64", "osx-64"]
   ```

2. **macOS-specific build tasks added:**
   ```toml
   [target.osx.tasks]
   build-ffi = "mojo build --emit shared-lib -I src -o libmojo_audio.dylib src/ffi/audio_ffi.mojo"
   build-ffi-optimized = "mojo build --emit shared-lib -O3 -I src -o libmojo_audio.dylib src/ffi/audio_ffi.mojo"
   ```

### GitHub Actions Workflow

New file: `.github/workflows/release.yml`

**Features:**
- Builds on `macos-latest` (Apple Silicon) and `macos-13` (Intel)
- Uses `prefix-dev/setup-pixi` action
- Packages releases with INSTALL.md instructions
- Creates GitHub Release with all platform artifacts

---

## Future Enhancements

Potential improvements for macOS support:

1. **Universal Binary:** Combine arm64 and x86_64 into single .dylib
2. **Homebrew Formula:** Distribute via Homebrew for easy installation
3. **Apple GPU Support:** Leverage Metal for GPU acceleration
4. **Codesigning:** Sign the dylib for distribution

---

## Resources

- [Mojo macOS Installation](https://docs.modular.com/mojo/manual/install/)
- [Pixi Multi-Platform Config](https://pixi.sh/reference/project_configuration/)
- [GitHub Actions macOS Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners)
