# Installation Guide

Complete installation guide for Futurnal on macOS, Windows, and Linux.

## Prerequisites

### Required: Ollama (Local LLM)

Futurnal uses Ollama for local LLM inference. Install it first:

**macOS**:
```bash
brew install ollama
```

**Windows**:
Download from [ollama.ai](https://ollama.ai) and run the installer.

**Linux**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

After installation, pull the required model:
```bash
ollama pull llama3.2:3b
```

### Optional: Better Models

For improved quality (requires more VRAM):
```bash
# 8B parameter model (recommended, 8GB VRAM)
ollama pull llama3.1:8b

# Qwen for reasoning tasks (optional)
ollama pull qwen2.5:7b
```

## Installation by Platform

### macOS

**Option 1: DMG Installer (Recommended)**
1. Download `Futurnal-1.0.0-arm64.dmg` (Apple Silicon) or `Futurnal-1.0.0-x64.dmg` (Intel)
2. Open the DMG file
3. Drag Futurnal to your Applications folder
4. First launch: Right-click > Open (to bypass Gatekeeper)

**Option 2: Homebrew**
```bash
brew install --cask futurnal
```

**Option 3: From Source (Python)**
```bash
# Clone repository
git clone https://github.com/futurnal/futurnal.git
cd futurnal

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Launch CLI
futurnal --help
```

### Windows

**Option 1: MSI Installer (Recommended)**
1. Download `Futurnal-1.0.0-x64.msi`
2. Run the installer
3. Follow the installation wizard
4. Launch from Start Menu

**Option 2: Portable (No install)**
1. Download `Futurnal-1.0.0-portable.zip`
2. Extract to desired location
3. Run `Futurnal.exe`

**Option 3: From Source (Python)**
```powershell
# Clone repository
git clone https://github.com/futurnal/futurnal.git
cd futurnal

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .

# Launch CLI
futurnal --help
```

### Linux

**Option 1: AppImage (Universal)**
```bash
# Download
wget https://github.com/futurnal/futurnal/releases/download/v1.0.0/Futurnal-1.0.0.AppImage

# Make executable
chmod +x Futurnal-1.0.0.AppImage

# Run
./Futurnal-1.0.0.AppImage
```

**Option 2: Debian/Ubuntu (.deb)**
```bash
# Download and install
wget https://github.com/futurnal/futurnal/releases/download/v1.0.0/futurnal_1.0.0_amd64.deb
sudo dpkg -i futurnal_1.0.0_amd64.deb
sudo apt-get install -f  # Install dependencies if needed
```

**Option 3: From Source (Python)**
```bash
# Clone repository
git clone https://github.com/futurnal/futurnal.git
cd futurnal

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Launch CLI
futurnal --help
```

## Verify Installation

### Check Ollama
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Check Futurnal
```bash
# CLI health check
futurnal health check

# Expected output:
# Futurnal Health Check
# ---------------------
# Ollama:     CONNECTED (localhost:11434)
# Models:     llama3.2:3b available
# Storage:    OK (2.1GB free)
# Privacy:    Consent registry initialized
```

## Configuration

Configuration is stored in `~/.futurnal/config.yaml`:

```yaml
# Futurnal Configuration

# LLM Settings
llm:
  provider: ollama
  endpoint: http://localhost:11434
  model: llama3.2:3b
  fallback_model: llama3.1:8b

# Storage
storage:
  data_dir: ~/.futurnal/data
  persist_path: ~/.futurnal/persist

# Privacy
privacy:
  audit_logging: true
  telemetry: false  # Opt-in only
  local_only: true  # No cloud by default

# Performance
performance:
  cache_enabled: true
  max_memory_gb: 2
  batch_size: 10
```

## First Launch

1. Launch Futurnal (desktop app or CLI)
2. Complete the onboarding wizard
3. Grant consent for data processing
4. Add your first data source (see [Data Sources](data-sources.md))

## Troubleshooting

### "Ollama not found"
Ensure Ollama is running:
```bash
ollama serve
```

### "Model not found"
Pull the required model:
```bash
ollama pull llama3.2:3b
```

### "Permission denied" (macOS)
First launch requires bypassing Gatekeeper:
- Right-click the app > Open > Open

### "Cannot connect to localhost:11434"
Check if Ollama is running and listening:
```bash
# Check if port is in use
lsof -i :11434

# Restart Ollama
ollama serve
```

## Updating

### Desktop App
The app will notify you of updates. Click "Update Now" to install.

### CLI/Python
```bash
cd futurnal
git pull
pip install -e .
```

## Uninstallation

### macOS
1. Move Futurnal.app to Trash
2. Remove data: `rm -rf ~/.futurnal`

### Windows
1. Control Panel > Programs > Uninstall Futurnal
2. Remove data: Delete `%USERPROFILE%\.futurnal`

### Linux
```bash
# AppImage
rm Futurnal-*.AppImage

# .deb
sudo dpkg -r futurnal

# Data
rm -rf ~/.futurnal
```

---

Next: [Quickstart Guide](quickstart.md)
