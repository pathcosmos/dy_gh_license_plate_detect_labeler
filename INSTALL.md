# Installation Guide

## Quick Installation

### For GPU systems (CUDA 12.1 - Latest):
```bash
# Install with uv (recommended)
uv pip install -r requirements-cuda121.txt

# Or with pip
pip install -r requirements-cuda121.txt
```

### For GPU systems (CUDA 12.0):
```bash
# Install with uv (recommended)
uv pip install -r requirements-cuda120.txt

# Or with pip
pip install -r requirements-cuda120.txt
```

### For GPU systems (CUDA 11.8 - Legacy):
```bash
# Install with uv (recommended)
uv pip install -r requirements-cuda118.txt

# Or with pip
pip install -r requirements-cuda118.txt
```

### For CPU-only systems:
```bash
# Install with uv (recommended)
uv pip install -r requirements-cpu.txt

# Or with pip
pip install -r requirements-cpu.txt
```

### Minimal Installation (for basic usage):
```bash
# Install with uv (recommended)
uv pip install -r requirements-minimal.txt

# Or with pip
pip install -r requirements-minimal.txt
```

## Development Installation

For development with additional tools:
```bash
# Install development dependencies
uv pip install -r requirements-dev.txt

# Or with pip
pip install -r requirements-dev.txt
```

## Verification

Test your installation:
```bash
# Check PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check Transformers installation
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Check Ultralytics installation
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"

# Check OpenCV installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## CUDA Support Check

Verify GPU support:
```bash
# Check CUDA availability and version
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Check GPU device information
python -c "import torch; print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Troubleshooting

If you encounter issues:

1. **CUDA version mismatch**:
   - Check your CUDA version: `nvidia-smi`
   - Use the appropriate requirements file for your CUDA version
   - For CUDA 12.1: `requirements-cuda121.txt`
   - For CUDA 12.0: `requirements-cuda120.txt`
   - For CUDA 11.8: `requirements-cuda118.txt`

2. **Memory issues**:
   - Try CPU-only installation: `requirements-cpu.txt`
   - Reduce batch size or image size
   - Use `--force-cpu` option when running the script

3. **Package conflicts**:
   - Create a fresh virtual environment
   - Use `requirements-minimal.txt` for basic functionality
   - Install additional packages as needed

4. **Installation errors**:
   - Ensure Python 3.8+ is installed
   - Update pip: `python -m pip install --upgrade pip`
   - Clear pip cache: `pip cache purge`
   - Try installing with `--no-cache-dir` option

## Virtual Environment Setup

Recommended setup with uv:
```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install with uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.8 or higher
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8 or higher
- 5GB+ free disk space

### Development Requirements
- All recommended requirements
- Additional 2GB disk space for development tools
- Git for version control
