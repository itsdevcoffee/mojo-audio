# macOS Release Build Guide - For Claude Agent

**Purpose:** Instructions for a Claude agent to build and upload macOS binaries for mojo-audio releases.

---

## Agent Task Overview

When a new release tag is created (e.g., `v0.1.2`), build the macOS binary and upload it to GitHub Releases.

---

## Prerequisites Check

Before starting, verify:

```bash
# Check tools are installed
command -v git >/dev/null 2>&1 || echo "❌ git not installed"
command -v pixi >/dev/null 2>&1 || echo "❌ pixi not installed"
command -v gh >/dev/null 2>&1 || echo "❌ gh CLI not installed"

# Check authentication
gh auth status || echo "❌ Not authenticated with GitHub"

# Check platform
uname -m  # Should show: arm64 (Apple Silicon) or x86_64 (Intel)
```

---

## Build Steps

### 1. Clone/Update Repository

```bash
# Navigate to workspace
cd ~/dev-coffee/repos  # Or wherever you keep projects

# Clone (first time only)
if [ ! -d "mojo-audio" ]; then
    git clone https://github.com/itsdevcoffee/mojo-audio.git
fi

cd mojo-audio

# Fetch latest tags
git fetch --tags

# Show available release tags
git tag -l "v*" | tail -5
```

### 2. Checkout Release Tag

```bash
# Replace v0.1.1 with the actual release version
RELEASE_TAG="v0.1.1"

git checkout $RELEASE_TAG

# Verify you're on the right tag
git describe --tags
```

### 3. Install Dependencies

```bash
# Clean previous build artifacts
rm -f libmojo_audio.dylib

# Install dependencies
pixi install

# Verify Mojo version
pixi run mojo --version
```

### 4. Build FFI Library

```bash
# Build optimized .dylib
pixi run build-ffi-optimized

# Verify build succeeded
if [ -f "libmojo_audio.dylib" ]; then
    echo "✅ Build successful"
    ls -lh libmojo_audio.dylib
    file libmojo_audio.dylib
else
    echo "❌ Build failed - libmojo_audio.dylib not found"
    exit 1
fi
```

**Expected output:**
- File size: ~20-30 KB
- File type: `Mach-O 64-bit dynamically linked shared library arm64` (or x86_64)

### 5. Package Release

```bash
# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    PLATFORM="macos-arm64"
elif [ "$ARCH" = "x86_64" ]; then
    PLATFORM="macos-x86_64"
else
    echo "❌ Unknown architecture: $ARCH"
    exit 1
fi

echo "Building for platform: $PLATFORM"

# Create release directory
mkdir -p release/$PLATFORM

# Copy files
cp libmojo_audio.dylib release/$PLATFORM/
cp include/mojo_audio.h release/$PLATFORM/

# Create installation guide
cat > release/$PLATFORM/INSTALL.md << EOF
# mojo-audio FFI - $PLATFORM

## Installation

\`\`\`bash
sudo cp libmojo_audio.dylib /usr/local/lib/
sudo cp mojo_audio.h /usr/local/include/
\`\`\`

## Verification

\`\`\`bash
ls -l /usr/local/lib/libmojo_audio.dylib
ls -l /usr/local/include/mojo_audio.h
\`\`\`

## Platform
- Architecture: $ARCH
- Built with: Mojo $(pixi run mojo --version 2>&1 | head -1)
EOF

# Create tarball
tar czf mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz -C release/$PLATFORM .

# Verify tarball
tar tzf mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz

echo "✅ Package created: mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz"
```

### 6. Upload to GitHub Release

```bash
# Check if release exists
gh release view $RELEASE_TAG >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Release $RELEASE_TAG does not exist on GitHub"
    echo "Create it first with: gh release create $RELEASE_TAG"
    exit 1
fi

# Upload asset
gh release upload $RELEASE_TAG mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz

echo "✅ Uploaded to: https://github.com/itsdevcoffee/mojo-audio/releases/tag/$RELEASE_TAG"
```

### 7. Verify Upload

```bash
# List release assets
gh release view $RELEASE_TAG --json assets --jq '.assets[].name'

# Should show:
# - mojo-audio-$RELEASE_TAG-linux-x86_64.tar.gz
# - mojo-audio-$RELEASE_TAG-macos-arm64.tar.gz (or macos-x86_64)
# - mojo-audio-ffi-examples.tar.gz
```

---

## Complete Automation Script

Save this as `build-macos-release.sh`:

```bash
#!/bin/bash
set -e  # Exit on error

RELEASE_TAG="${1:-v0.1.1}"

echo "🔨 Building macOS release for $RELEASE_TAG"

# Navigate to repo
cd ~/dev-coffee/repos/mojo-audio || exit 1

# Checkout release
git fetch --tags
git checkout $RELEASE_TAG

# Clean and build
rm -f libmojo_audio.dylib
pixi install
pixi run build-ffi-optimized

# Detect architecture
ARCH=$(uname -m)
PLATFORM="macos-$ARCH"

# Package
mkdir -p release/$PLATFORM
cp libmojo_audio.dylib release/$PLATFORM/
cp include/mojo_audio.h release/$PLATFORM/
cat > release/$PLATFORM/INSTALL.md << EOF
# mojo-audio FFI - $PLATFORM
## Installation
\`\`\`bash
sudo cp libmojo_audio.dylib /usr/local/lib/
sudo cp mojo_audio.h /usr/local/include/
\`\`\`
EOF

tar czf mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz -C release/$PLATFORM .

# Upload
gh release upload $RELEASE_TAG mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz --clobber

echo "✅ Success! Uploaded mojo-audio-$RELEASE_TAG-$PLATFORM.tar.gz"
echo "🔗 https://github.com/itsdevcoffee/mojo-audio/releases/tag/$RELEASE_TAG"
```

**Usage:**
```bash
chmod +x build-macos-release.sh
./build-macos-release.sh v0.1.2  # Replace with actual version
```

---

## Troubleshooting

### Build Errors

**Error:** `unable to locate module 'types'`
- This is a known issue with newer Mojo versions
- Try downgrading: The build should work with Mojo 0.26.1 dev builds from January 2026
- Or wait for Mojo API stabilization

**Error:** `'ImmutOrigin' value has no attribute 'external'`
- Code needs updating for current Mojo API
- Check if there's a fix in main branch
- May need to use an older Mojo version temporarily

### Upload Errors

**Error:** `release not found`
```bash
# Create the release first
gh release create v0.1.2 --title "v0.1.2" --notes "See CHANGELOG.md"
```

**Error:** `HTTP 403: Resource protected by organization`
```bash
# Re-authenticate
gh auth login
```

---

## Success Checklist

- [ ] Built `libmojo_audio.dylib` successfully
- [ ] Verified file is Mach-O format for correct architecture
- [ ] Created tarball with .dylib, .h, and INSTALL.md
- [ ] Uploaded to GitHub release
- [ ] Verified download link works
- [ ] Tested installation on clean macOS system (optional but recommended)

---

## Notes for Agent

- Always verify the release tag exists before starting
- Check GitHub release page after upload to confirm
- If build fails due to Mojo API issues, report to user - don't try to fix code
- Keep build logs for debugging
- Each release should have both arm64 and x86_64 builds if possible (requires access to both platforms)

---

**Last Updated:** February 4, 2026
**Maintained for:** Claude agents running on macOS
