# Manual macOS Build for v0.1.1 Release

**Date:** February 4, 2026
**Purpose:** Create macOS binaries for v0.1.1 release

---

## Prerequisites on Your MacBook

- macOS Ventura (13) or later
- Xcode Command Line Tools installed
- Homebrew installed
- Pixi installed

---

## Step-by-Step Build Instructions

### 1. Clone and Setup

```bash
# Clone the repository (or pull if you already have it)
cd ~/dev-coffee/repos  # or wherever you keep your projects
git clone https://github.com/itsdevcoffee/mojo-audio.git
cd mojo-audio

# Checkout the v0.1.1 tag
git fetch --tags
git checkout v0.1.1

# Install dependencies
pixi install
```

### 2. Verify Mojo Version

```bash
# Check that you got the stable version
pixi run mojo --version

# Should show: 0.26.1.0 (release) or similar
```

### 3. Build the FFI Library

```bash
# Build optimized .dylib for macOS
pixi run build-ffi-optimized

# Verify the build
ls -lh libmojo_audio.dylib
file libmojo_audio.dylib

# Expected output should show:
# - File size around 20-30KB
# - Mach-O 64-bit dynamically linked shared library arm64
```

### 4. Package for Release

```bash
# Create release directory
mkdir -p release/macos-arm64

# Copy files
cp libmojo_audio.dylib release/macos-arm64/
cp include/mojo_audio.h release/macos-arm64/

# Create installation instructions
cat > release/macos-arm64/INSTALL.md << 'EOF'
# mojo-audio FFI - macOS Apple Silicon (arm64)

## Installation

```bash
sudo cp libmojo_audio.dylib /usr/local/lib/
sudo cp mojo_audio.h /usr/local/include/
```

## Verification

```bash
ls -l /usr/local/lib/libmojo_audio.dylib
ls -l /usr/local/include/mojo_audio.h
```

## Usage

Link against the library when compiling:

```bash
# C
gcc -o demo demo.c -lmojo_audio

# Rust
# Add to your build.rs or link manually
```

## Requirements

- macOS Ventura (13) or later
- Apple Silicon (M1/M2/M3)
EOF

# Create tarball
tar czf mojo-audio-v0.1.1-macos-arm64.tar.gz -C release/macos-arm64 .

# Verify tarball
tar tzf mojo-audio-v0.1.1-macos-arm64.tar.gz
```

### 5. Upload to GitHub Release

```bash
# Option A: Using gh CLI (recommended)
gh release upload v0.1.1 mojo-audio-v0.1.1-macos-arm64.tar.gz

# Option B: Manual upload via GitHub web interface
# 1. Go to: https://github.com/itsdevcoffee/mojo-audio/releases/tag/v0.1.1
# 2. Click "Edit release"
# 3. Drag and drop: mojo-audio-v0.1.1-macos-arm64.tar.gz
# 4. Click "Update release"
```

---

## Optional: Build for Intel Macs (x86_64)

If you want to also build for Intel Macs, you'll need an Intel Mac or use Rosetta:

```bash
# Note: This may not work perfectly via Rosetta, but you can try
arch -x86_64 pixi run build-ffi-optimized

# Then package similarly:
mkdir -p release/macos-x86_64
cp libmojo_audio.dylib release/macos-x86_64/
cp include/mojo_audio.h release/macos-x86_64/
# ... (same packaging steps as above)
tar czf mojo-audio-v0.1.1-macos-x86_64.tar.gz -C release/macos-x86_64 .
```

---

## Troubleshooting

### Build fails with "unable to locate module 'types'"

```bash
# Verify the source structure
tree src/ffi/

# Should show:
# src/ffi/
# ├── __init__.mojo
# ├── audio_ffi.mojo
# └── types.mojo

# If missing, you're not on the right branch/tag
git status
git checkout v0.1.1
```

### "ImmutOrigin" error

```bash
# This means you have an old Mojo version
pixi run mojo --version

# Should be 0.26.1.0 (release)
# If not, run:
pixi update mojo max
```

### Permission denied when uploading

```bash
# Make sure you're authenticated
gh auth status

# If not logged in:
gh auth login
```

---

## Success Checklist

- [ ] Built `libmojo_audio.dylib` successfully
- [ ] Verified file is Mach-O arm64 format
- [ ] Created tarball with .dylib, .h, and INSTALL.md
- [ ] Uploaded to GitHub release v0.1.1
- [ ] Verified download link works

---

## Next Steps

After uploading:
1. Update the release notes to mention macOS support
2. Test the download and installation on a clean macOS system
3. Announce macOS availability

---

## Future: Automated macOS Builds

For v0.1.2+, the GitHub Actions workflow will handle this automatically:
- Updated to stable Mojo 26.1.0
- macOS builds will work in CI
- No manual building needed

This manual process is only needed for v0.1.1 since we discovered the version
mismatch issue after releasing.
