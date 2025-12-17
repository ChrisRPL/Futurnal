#!/bin/bash
# =============================================================================
# Futurnal Cross-Platform Build Script
# =============================================================================
#
# Builds Futurnal for all supported platforms:
# - macOS (arm64, x64)
# - Windows (x64)
# - Linux (x64)
#
# Prerequisites:
# - Node.js 20+
# - Rust stable
# - Platform-specific dependencies
#
# Usage:
#   ./scripts/build-all-platforms.sh [options]
#
# Options:
#   --version VERSION   Set build version (default: from package.json)
#   --platform PLATFORM Build for specific platform (macos, windows, linux, all)
#   --sign              Sign builds (requires signing keys)
#   --verbose           Enable verbose output
#   --help              Show this help message
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DESKTOP_DIR="$PROJECT_ROOT/desktop"

# Default values
VERSION=""
PLATFORM="all"
SIGN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --sign)
            SIGN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --version VERSION   Set build version"
            echo "  --platform PLATFORM Build for platform (macos, windows, linux, all)"
            echo "  --sign              Sign builds"
            echo "  --verbose           Enable verbose output"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    NODE_VERSION=$(node --version)
    log_info "Node.js: $NODE_VERSION"

    # npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
        exit 1
    fi
    NPM_VERSION=$(npm --version)
    log_info "npm: $NPM_VERSION"

    # Rust
    if ! command -v rustc &> /dev/null; then
        log_error "Rust is not installed"
        exit 1
    fi
    RUST_VERSION=$(rustc --version)
    log_info "Rust: $RUST_VERSION"

    # Cargo
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo is not installed"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get version from package.json if not specified
get_version() {
    if [[ -z "$VERSION" ]]; then
        VERSION=$(node -p "require('$DESKTOP_DIR/package.json').version")
    fi
    log_info "Build version: $VERSION"
}

# Update version in files
update_version() {
    log_info "Updating version to $VERSION..."

    # Update package.json
    cd "$DESKTOP_DIR"
    npm version "$VERSION" --no-git-tag-version --allow-same-version 2>/dev/null || true

    # Update tauri.conf.json
    cd "$DESKTOP_DIR/src-tauri"
    if command -v jq &> /dev/null; then
        jq ".version = \"$VERSION\"" tauri.conf.json > tauri.conf.json.tmp
        mv tauri.conf.json.tmp tauri.conf.json
    else
        log_warning "jq not found, skipping tauri.conf.json update"
    fi

    log_success "Version updated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."

    cd "$DESKTOP_DIR"
    npm ci

    log_success "Dependencies installed"
}

# Build for macOS
build_macos() {
    log_info "Building for macOS..."

    cd "$DESKTOP_DIR"

    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_warning "Not running on macOS, skipping macOS build"
        log_warning "Cross-compilation for macOS requires native macOS"
        return
    fi

    # Build for ARM64 (Apple Silicon)
    log_info "Building for macOS ARM64..."
    if [[ "$SIGN" == true ]]; then
        npm run tauri build -- --target aarch64-apple-darwin
    else
        npm run tauri build -- --target aarch64-apple-darwin
    fi

    # Build for x64 (Intel)
    log_info "Building for macOS x64..."
    npm run tauri build -- --target x86_64-apple-darwin

    log_success "macOS builds complete"
}

# Build for Windows
build_windows() {
    log_info "Building for Windows..."

    cd "$DESKTOP_DIR"

    # Check if running on Windows or has cross-compilation
    if [[ "$(uname)" == "MINGW"* ]] || [[ "$(uname)" == "MSYS"* ]] || [[ "$(uname)" == "CYGWIN"* ]]; then
        npm run tauri build -- --target x86_64-pc-windows-msvc
        log_success "Windows build complete"
    else
        log_warning "Not running on Windows"
        log_warning "Cross-compilation for Windows requires Windows or Wine"
    fi
}

# Build for Linux
build_linux() {
    log_info "Building for Linux..."

    cd "$DESKTOP_DIR"

    # Check if running on Linux
    if [[ "$(uname)" != "Linux" ]]; then
        log_warning "Not running on Linux, skipping Linux build"
        return
    fi

    # Check for required dependencies
    if ! dpkg -l | grep -q libwebkit2gtk-4.1-dev; then
        log_error "libwebkit2gtk-4.1-dev is required for Linux build"
        log_info "Install with: sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf"
        exit 1
    fi

    npm run tauri build -- --target x86_64-unknown-linux-gnu

    log_success "Linux build complete"
}

# Build Python package
build_python() {
    log_info "Building Python package..."

    cd "$PROJECT_ROOT"

    # Check for build tools
    if ! python -m pip show build &> /dev/null; then
        log_info "Installing build tools..."
        python -m pip install build
    fi

    # Update version in pyproject.toml
    if [[ -f "pyproject.toml" ]]; then
        sed -i.bak "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
        rm -f pyproject.toml.bak
    fi

    # Build
    python -m build

    log_success "Python package built"
}

# Generate checksums
generate_checksums() {
    log_info "Generating checksums..."

    cd "$DESKTOP_DIR/src-tauri/target"

    # Find all build artifacts
    find . -type f \( -name "*.dmg" -o -name "*.msi" -o -name "*.exe" -o -name "*.AppImage" -o -name "*.deb" \) -exec sha256sum {} \; > checksums.txt

    if [[ -s checksums.txt ]]; then
        log_success "Checksums generated:"
        cat checksums.txt
    else
        log_warning "No build artifacts found for checksums"
    fi
}

# Main build process
main() {
    echo "============================================"
    echo "  Futurnal Cross-Platform Build"
    echo "============================================"
    echo ""

    check_prerequisites
    get_version
    update_version
    install_dependencies

    case "$PLATFORM" in
        macos)
            build_macos
            ;;
        windows)
            build_windows
            ;;
        linux)
            build_linux
            ;;
        python)
            build_python
            ;;
        all)
            build_macos
            build_windows
            build_linux
            build_python
            ;;
        *)
            log_error "Unknown platform: $PLATFORM"
            exit 1
            ;;
    esac

    generate_checksums

    echo ""
    echo "============================================"
    log_success "Build complete!"
    echo "============================================"
    echo ""
    echo "Build artifacts located in:"
    echo "  Desktop: $DESKTOP_DIR/src-tauri/target/"
    echo "  Python:  $PROJECT_ROOT/dist/"
    echo ""
}

# Run main
main
