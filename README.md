# License Plate Detection and YOLO Labeling Generator

An AI-powered tool that automatically detects license plates in images and generates YOLO format label files using state-of-the-art deep learning models.

## 🚀 Key Features

- **Automatic License Plate Detection**: Accurate license plate detection using various models (YOLOS, DETR, YOLOv5, YOLOv8)
- **YOLO Label Generation**: Automatic conversion of detected license plates to YOLO format labels
- **Image Size Optimization**: Automatic processing size adjustment based on input image dimensions
- **Visualization Support**: Generate visual images to verify detection results
- **Batch Processing**: Process all images in a directory at once
- **Undetected Image Management**: Separate storage for images where no license plates were detected
- **GPU Acceleration**: Automatic GPU detection and usage for faster processing
- **Flexible Device Control**: Option to force CPU usage when needed

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Package Management Options

#### Option 1: Using uv (Recommended for fast installation)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies with uv
uv pip install torch torchvision
uv pip install transformers
uv pip install opencv-python
uv pip install Pillow
uv pip install numpy
uv pip install ultralytics
```

Or install from requirements.txt:
```bash
uv pip install -r requirements.txt
```

#### Option 2: Using pip (Traditional method)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install Pillow
pip install numpy
pip install ultralytics
```

Or if you have a requirements.txt file:
```bash
pip install -r requirements.txt
```

### CUDA Support

For GPU acceleration, install the appropriate CUDA version:

#### CUDA 12.1 (Latest)
```bash
uv pip install -r requirements-cuda121.txt
# or
pip install -r requirements-cuda121.txt
```

#### CUDA 12.0
```bash
uv pip install -r requirements-cuda120.txt
# or
pip install -r requirements-cuda120.txt
```

#### CUDA 11.8 (Legacy)
```bash
uv pip install -r requirements-cuda118.txt
# or
pip install -r requirements-cuda118.txt
```

#### CPU-only Installation
```bash
uv pip install -r requirements-cpu.txt
# or
pip install -r requirements-cpu.txt
```

### Verification

After installation, verify your setup:
```bash
# Check PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check Transformers installation
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Check Ultralytics installation
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"

# Check OpenCV installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🛠️ Installation

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

### **Clone Repository**
```bash
git clone <repository-url>
cd <downloaded-git-folder>
```

### Quick Installation

#### Option 1: Using uv (Recommended for fast installation)
```bash
# Create virtual environment with uv
uv venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

#### Using traditional venv
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. **Install Dependencies**

#### With uv (Recommended)
```bash
uv pip install -r requirements.txt
```

#### With pip
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

**Single Image Processing:**
```bash
python src/license_plate_labeler.py -i path/to/image.jpg -o output_directory
```

**Directory Batch Processing:**
```bash
python src/license_plate_labeler.py -i input_directory -o output_directory
```

### Advanced Options

**Adjust Confidence Threshold:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.7
```

**Disable Visualization:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --no-viz
```

**Save Undetected Images:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir -e undetected_folder
```

**Set Maximum Processing Size:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 1024
```

**Force CPU Usage:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --force-cpu
```

**Combination of All Options:**
```bash
python src/license_plate_labeler.py \
  -i data/images \
  -o results \
  -c 0.6 \
  --max-size 1024 \
  -e undetected \
  --no-viz \
  --force-cpu
```

## 📝 Command Line Options

| Option | Description |
|--------|-------------|
| `--input` or `-i` | Input image file or directory path (Required) |
| `--output` or `-o` | Output directory path (Required) |
| `--model` | Model to use for detection (default: yolos-small) |
| | *(YOLOv11 모델은 처음 사용 시 Hugging Face에서 자동으로 다운로드되어 `~/.cache/license_plate_models` 폴더에 저장됩니다.)*
| `--confidence` or `-c` | Confidence threshold (default 0.5) |
| `--no-viz` | Disable visualization output (default False) |
| `-e` | Directory to save undetected images (default None) |
| `--max-size` | Maximum processing size (default 800) |
| `--force-cpu` | Force CPU usage (default False) |
| `--token` or `-t` | Hugging Face API token for model download (use read-only token) |
| `--local-model` | Path to local model file for offline use |
| `--list-models` | List all available models and exit |

## Available Models

| Model | Framework | Size | Accuracy (mAP@50) | Speed | GPU Memory | CPU Usage |
|-------|-----------|------|-------------------|-------|------------|-----------|
| yolov11x | YOLOv11 | 114MB | 0.9813 | ⭐ | 8GB+ | ❌ |
| yolov11l | YOLOv11 | 51.2MB | 0.9789 | ⭐⭐ | 6GB+ | ❌ |
| yolov11m | YOLOv11 | 40.5MB | 0.9765 | ⭐⭐⭐ | 4GB+ | ❌ |
| detr-resnet-50 | DETR | 159MB | 0.9756 | ⭐⭐ | 6GB+ | ❌ |
| yolov11s | YOLOv11 | 19.2MB | 0.9742 | ⭐⭐⭐⭐ | 2GB+ | ⚠️ |
| yolos-small-finetuned | YOLOS | 90MB | 0.9731 | ⭐⭐⭐ | 2GB+ | ⚠️ |
| yolos-rego | YOLOS | 90MB | 0.9728 | ⭐⭐⭐ | 2GB+ | ⚠️ |
| yolov11n | YOLOv11 | 5.47MB | 0.9715 | ⭐⭐⭐⭐⭐ | 1GB+ | ✅ |
| yolov5m | YOLOv5 | 40MB | 0.9702 | ⭐⭐⭐⭐⭐ | 1GB+ | ✅ |
| yolov8-lp-mkgoud | YOLOv8 | 6.24MB | 0.9698 | ⭐⭐⭐⭐ | 1GB+ | ✅ |

## 📁 Output File Structure

After processing, the following files will be generated in the output directory:

```
output_directory/
├── image1.jpg                    # Original image (copied from input)
├── image1.txt                    # YOLO label file (same name as image)
├── image1_detected.jpg           # Detection visualization (optional)
├── image2.png                    # Original image (copied from input)
├── image2.txt                    # YOLO label file (same name as image)
├── image2_detected.jpg           # Detection visualization (optional)
└── ...
```

**Key Features of File Organization:**
- **Original Images**: Source images are copied to the output directory preserving their original filenames and extensions
- **Label Files**: YOLO label files (.txt) have identical filenames to their corresponding images, differing only in extension
- **Visualization Files**: Detection visualization images use the `_detected.jpg` suffix for easy identification
- **Perfect Dataset Structure**: Ready-to-use YOLO training dataset with properly paired images and labels

### YOLO Label File Format

Each `.txt` file follows this format:
```
class_id x_center y_center width height
```

Example:
```
0 0.512500 0.345833 0.125000 0.158333
0 0.687500 0.512500 0.156250 0.145833
```

- `class_id`: Class ID (license plate is 0)
- `x_center, y_center`: Bounding box center point (normalized coordinates, 0.0-1.0)
- `width, height`: Bounding box size (normalized size, 0.0-1.0)

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## ⚙️ Performance Optimization

### Image Size Optimization
The program automatically analyzes input image size to determine optimal processing size:

- **Large Images (>800px)**: Downscale to maximum size for improved processing speed
- **Small Images (<400px)**: Upscale to minimum size for better detection accuracy
- **Appropriate Size**: Adjust to multiples of 8 for model efficiency optimization

### GPU Usage
When CUDA is installed, the program automatically detects and uses GPU to improve processing speed. The program will display detailed GPU information including:
- GPU name and device count
- Memory usage statistics
- Automatic fallback to CPU if GPU is unavailable

### Device Selection
- **Automatic GPU Detection**: Uses GPU automatically when CUDA is available
- **Manual CPU Override**: Use `--force-cpu` to disable GPU usage
- **Memory Management**: Automatic GPU memory monitoring and error handling

### Memory Optimization
- Large images are automatically resized to reduce memory usage
- During batch processing, images are processed one by one to prevent memory overflow

## 🔧 Troubleshooting

### Common Errors

**1. Model Loading Failure**
```
Model loading failed: HTTPSConnectionPool...
```
**Solution**: 
- Check internet connection
- Check firewall settings
- Try disconnecting VPN

**2. GPU Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: 
```bash
# Reduce image processing size
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 600

# Or force CPU usage
python src/license_plate_labeler.py -i input_dir -o output_dir --force-cpu
```

**3. GPU Not Detected**
```
CUDA를 사용할 수 없습니다. CPU를 사용합니다.
```
**Solution**: 
- Verify NVIDIA GPU is installed
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support if needed

**4. No Image Files Found**
```
No image files found in input directory
```
**Solution**: 
- Check input path
- Verify supported file extensions
- Check file permissions

**5. Permission Error**
```
Permission denied
```
**Solution**: 
- Check write permissions for output directory
- May require `sudo` privileges

**6. Dependency Error**
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution**: 
```bash
pip install -r requirements.txt
# or with uv
uv pip install -r requirements.txt
```

**7. Size Parameter Error**
```
ValueError: Size must contain 'height' and 'width' keys...
```
**Solution**: 
- Update transformers library to latest version
- The program automatically handles compatibility

### Model Access Issues

If you encounter errors like:
```
valentinafeve/yolos-small_finetuned_license_plate is not a local folder and is not a valid model identifier
```

**Solutions:**

1. **Use the recommended model** (most reliable):
   ```bash
   python src/license_plate_labeler.py -i input_dir -o output_dir -m yolos-small
   ```

2. **For private/restricted models**, authenticate with HuggingFace:

   #### Method 1: Interactive Login (Recommended)
   ```bash
   # Install HuggingFace CLI if not already installed
   pip install huggingface_hub
   
   # Login to HuggingFace (opens web browser for authentication)
   huggingface-cli login
   ```
   This will:
   - Open a web browser to HuggingFace login page
   - Ask you to create or paste an access token
   - Store the token locally for future use

   #### Method 2: Direct Token Authentication
   ```bash
   # Set token as environment variable
   export HF_TOKEN="your_token_here"
   
   # Or on Windows
   set HF_TOKEN=your_token_here
   ```

   #### Method 3: Programmatic Token Usage
   If you need to pass the token directly in your code, you can modify the initialization:
   ```python
   # Add token parameter to model loading (for advanced users)
   from transformers import YolosImageProcessor, YolosForObjectDetection
   
   processor = YolosImageProcessor.from_pretrained(
       model_name, 
       token="your_token_here"
   )
   model = YolosForObjectDetection.from_pretrained(
       model_name, 
       token="your_token_here"
   )
   ```

   #### Getting HuggingFace Access Token
   1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
   2. Click "Create new token"
   3. Choose appropriate permissions:
      - **Read**: For downloading public models
      - **Write**: If you plan to upload models
      - For this project a **Read** token is enough
   4. Copy the generated token
   5. Use it with one of the methods above

   #### Token Security Best Practices
   - **Never commit tokens to version control**
   - **Use environment variables for tokens in production**
   - **Regenerate tokens if compromised**
   - **Use read-only tokens when possible**

3. **Check model availability**:
   - Visit the model page directly: https://huggingface.co/valentinafeve/yolos-small_finetuned_license_plate
   - Some models may be private or moved
   - Use `--list-models` to see all available options

4. **Use alternative models**:
   ```bash
   # Try the base YOLOS model
   python src/license_plate_labeler.py -i input_dir -o output_dir -m yolos-base
   
   # Or DETR ResNet-50
   python src/license_plate_labeler.py -i input_dir -o output_dir -m detr-resnet-50
   ```

### Authentication Troubleshooting

#### Common Authentication Issues

| Error Message | Solution |
|---------------|----------|
| `401 Client Error: Unauthorized` | Login required: `huggingface-cli login` |
| `403 Client Error: Forbidden` | Token lacks permissions or model is private |
| `Token is not valid` | Generate new token from HuggingFace settings |
| `Repository not found` | Check model name or verify access permissions |

#### Verifying Authentication
```bash
# Check if you're logged in
huggingface-cli whoami

# Test token validity
huggingface-cli repo info microsoft/DialoGPT-medium

# List your accessible models (if any)
huggingface-cli list-models --author your-username
```

#### Logout and Re-authentication
```bash
# Logout from HuggingFace CLI
huggingface-cli logout

# Login again
huggingface-cli login
```

## 📈 Performance Tuning Guide

### Improve Detection Accuracy
```bash
# Lower confidence threshold
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.3

# Increase processing size
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 1024

# Combination
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.3 --max-size 1024
```

### Improve Processing Speed
```bash
# Higher confidence threshold
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.8

# Reduce processing size
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 600

# Disable visualization
python src/license_plate_labeler.py -i input_dir -o output_dir --no-viz

# Maximum speed combination
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.8 --max-size 600 --no-viz
```

### GPU and Performance Optimization
```bash
# Force CPU usage for compatibility
python src/license_plate_labeler.py -i input_dir -o output_dir --force-cpu

# GPU with memory optimization
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 600

# Maximum GPU performance
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 1024
```

### Batch Processing Optimization
```bash
# Save undetected images for post-processing efficiency
python src/license_plate_labeler.py -i input_dir -o output_dir -e undetected

# Memory-constrained environments
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 400
```

## 📊 Real-World Usage Examples

### Example 1: Basic Processing
```bash
python src/license_plate_labeler.py -i data/cars -o results
```
**Use Case**: General license plate detection and labeling

**Output Structure**:
```
results/
├── car001.jpg          # Original image (copied)
├── car001.txt          # YOLO label file
├── car001_detected.jpg # Visualization image
├── car002.png          # Original image (copied)
├── car002.txt          # YOLO label file
├── car002_detected.jpg # Visualization image
└── ...
```

### Example 2: High-Precision Detection
```bash
python src/license_plate_labeler.py \
  -i data/difficult_cases \
  -o results \
  -c 0.3 \
  --max-size 1024 \
  -e undetected
```
**Use Case**: Processing images with small license plates or complex backgrounds

**Features**:
- Low confidence for more candidate detection
- Large processing size to preserve details
- Separate storage for undetected images

### Example 3: Fast Bulk Processing
```bash
python src/license_plate_labeler.py \
  -i data/large_dataset \
  -o results \
  -c 0.8 \
  --max-size 600 \
  --no-viz
```
**Use Case**: Processing thousands of images quickly

**Features**:
- High confidence for certain detections only
- Small processing size for speed improvement
- Skip visualization to save storage time

### Example 4: Quality Inspection
```bash
python src/license_plate_labeler.py \
  -i data/test_images \
  -o results \
  -c 0.5 \
  -e undetected
```
**Use Case**: Performance evaluation and undetected image analysis

**Features**:
- Default confidence for balanced detection
- Undetected images for model performance analysis

### Example 5: Production Environment
```bash
python src/license_plate_labeler.py \
  -i /mnt/input_images \
  -o /mnt/output_labels \
  -c 0.6 \
  --max-size 800 \
  --no-viz \
  >> /var/log/license_plate_processing.log 2>&1
```
**Use Case**: Stable batch processing in server environment

**Features**:
- Log file for processing record management
- Stable confidence setting
- Disabled visualization for resource conservation

### Example 6: CPU-Only Processing
```bash
python src/license_plate_labeler.py \
  -i data/images \
  -o results \
  -c 0.5 \
  --force-cpu
```
**Use Case**: Processing on systems without GPU or when GPU memory is limited

**Features**:
- Guaranteed CPU processing
- No GPU memory requirements
- Consistent performance across different hardware

## 🎯 Best Practices

### Data Preparation
1. **Image Quality**
   - Resolution: Minimum 640x480 recommended
   - Format: Use JPEG or PNG
   - Compression: Avoid excessive compression

2. **File Management**
   - Filenames: Use alphanumeric characters
   - Directory structure: Organized folder structure
   - Backup: Original files are automatically copied to output directory
   - **Dataset Ready**: Output directory contains complete YOLO dataset structure

3. **Preprocessing**
   - Image rotation: Adjust so license plates are horizontal
   - Brightness/contrast: Maintain appropriate brightness and contrast
   - Noise removal: Remove noise when possible

### Processing Optimization
1. **Device Settings**
   - GPU recommended: Automatic detection and usage
   - CPU fallback: Automatic when GPU unavailable
   - Manual override: Use `--force-cpu` when needed

2. **Output Management**
   - **Complete Dataset**: Images and labels are organized together in output directory
   - **Filename Matching**: Label files exactly match image filenames (except extension)
   - **Training Ready**: Output can be directly used for YOLO model training

### Result Validation
1. **Visualization Check**
   - Check visualization results for first few images
   - Review bounding box accuracy
   - Check confidence score distribution

2. **Dataset Integrity Verification**
   ```bash
   # Check that every image has a corresponding label file
   cd output_directory
   
   # Count images (excluding visualization files)
   ls *.jpg *.png *.jpeg *.bmp *.tiff 2>/dev/null | grep -v "_detected" | wc -l
   
   # Count label files
   ls *.txt | wc -l
   
   # These counts should match
   
   # Verify specific file pairing
   for img in *.jpg *.png *.jpeg *.bmp *.tiff; do
       if [[ "$img" != *"_detected"* ]]; then
           base="${img%.*}"
           if [[ ! -f "${base}.txt" ]]; then
               echo "Missing label for: $img"
           fi
       fi
   done
   ```

3. **Label Format Validation**
   ```python
   # Validate YOLO label format
   import os
   import glob
   
   label_dir = "output_directory"
   for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
       with open(label_file, 'r') as f:
           for line_num, line in enumerate(f, 1):
               parts = line.strip().split()
               if len(parts) != 5:
                   print(f"Invalid format in {label_file}:{line_num}")
               else:
                   try:
                       class_id, x, y, w, h = map(float, parts)
                       if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                           print(f"Coordinate out of range in {label_file}:{line_num}")
                   except ValueError:
                       print(f"Invalid values in {label_file}:{line_num}")
   ```

## 🤝 Contributing

### Bug Reports
Please report bugs through GitHub Issues. Include the following information:
- Operating system and Python version
- Command used
- Complete error message
- Reproducible steps

### Feature Suggestions
Please suggest new features or improvements:
- Describe specific use cases
- Expected implementation direction
- Related references or papers

### Code Contributions
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- **Coding Style**: Follow PEP 8
- **Documentation**: Write docstrings for functions and classes
- **Testing**: Include test code for new features
- **Compatibility**: Maintain Python 3.8+ support
- **Package Management**: Support both pip and uv for dependency installation

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for details.

### MIT License Key Points
- ✅ **Commercial Use Allowed**: Unrestricted commercial use
- ✅ **Modification and Distribution**: Code modification and redistribution allowed
- ✅ **Private Use**: Free personal use
- ❗ **License Notice Required**: License and copyright notice required

## 📞 Support and Community

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Wiki**: Additional documentation and tutorials

### Related Resources
- [YOLO Official Documentation](https://docs.ultralytics.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyTorch Official Documentation](https://pytorch.org/docs/)

## 🙏 Acknowledgments

### Open Source Projects
- [HuggingFace Transformers](https://huggingface.co/transformers/) - YOLOS model provider
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [Pillow](https://pillow.readthedocs.io/) - Image processing library

### Pre-trained Models
- [nickmuchi/yolos-small-finetuned-license-plate-detection](https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection) - License plate detection model

### Contact
- **Author**: lanco.gh@gmail.com
- **GitHub Issues**: For bug reports and feature requests

### Contributors
Thanks to everyone who has contributed to this project.

---

## 📝 Changelog

### v1.0.0 (Current)
- ✅ Initial release
- ✅ Basic license plate detection functionality
- ✅ YOLO label generation functionality
- ✅ Batch processing support
- ✅ Image size optimization
- ✅ Undetected image management
- ✅ Support for both pip and uv package managers
- ✅ Automatic GPU detection and usage
- ✅ CPU fallback and force CPU option
- ✅ GPU memory monitoring and optimization
- ✅ Original image copying to output directory
- ✅ Perfect filename matching between images and labels

### v1.1.0 (Planned)
- 🔄 Performance optimization
- 🔄 Additional model support
- 🔄 Web interface provision
- 🔄 Dataset management tools

---

## 📈 Performance Metrics

### Test Environment
- **CPU**: Intel i5-13500
- **GPU**: NVIDIA RTX 4070
- **RAM**: 32GB DDR5
- **Images**: 1600x1200 JPEG

### Processing Speed
- **CPU Processing**: ~2-3 seconds/image
- **GPU Processing**: ~0.5-1 seconds/image (varies by GPU)
- **Batch Processing**: ~1000 images/hour (GPU), ~300 images/hour (CPU)

### GPU Requirements
- **Minimum VRAM**: 4GB recommended
- **Optimal VRAM**: 8GB+ for large images
- **CUDA Version**: Compatible with PyTorch installation

---

**If you encounter any issues or have questions, please feel free to contact us through GitHub Issues!**

**Happy License Plate Detection! 🚗📋**

## 📊 Available Models

### 🔥 최고 정확도 모델
- `yolov11x`: YOLOv11 Extra Large (mAP@50: 0.9813)
  - 가장 높은 정확도를 제공하는 대형 모델
  - 고사양 GPU 필요

### 💪 고성능 모델
- `yolov11l`: YOLOv11 Large (mAP@50: 0.9789)
  - 높은 정확도와 적절한 속도 균형
  - 중간~고사양 GPU 권장
- `yolov11m`: YOLOv11 Medium (mAP@50: 0.9765)
  - 균형잡힌 성능과 속도
  - 중간 사양 GPU 권장
- `detr-resnet50`: DETR with ResNet-50 (mAP@50: 0.9756)
  - Transformer 기반 모델
  - 중간 사양 GPU 권장

### ⚖️ 균형잡힌 모델
- `yolov11s`: YOLOv11 Small (mAP@50: 0.9742)
  - 가벼우면서도 좋은 성능
  - 저사양 GPU에서도 사용 가능
- `yolos-small`: YOLOS Small (mAP@50: 0.9731)
  - Transformer 기반 경량 모델
  - 저사양 GPU에서도 사용 가능
- `yolos-rego`: YOLOS Rego (mAP@50: 0.9728)
  - 특화된 경량 Transformer 모델
  - 저사양 GPU에서도 사용 가능

### 🚀 경량 모델
- `yolov11n`: YOLOv11 Nano (mAP@50: 0.9715)
  - 가장 가벼운 YOLOv11 모델
  - CPU에서도 사용 가능
- `yolov5m`: YOLOv5 Medium (mAP@50: 0.9702)
  - 검증된 안정적인 모델
  - CPU에서도 사용 가능
- `yolov8s`: YOLOv8 Small (mAP@50: 0.9698)
  - 최신 YOLOv8 기반 경량 모델
  - CPU에서도 사용 가능

### 모델 선택 가이드

1. **최고 정확도가 필요한 경우**
   - `yolov11x` 사용
   - 고사양 GPU 필요

2. **균형잡힌 성능이 필요한 경우**
   - `yolov11m` 또는 `yolov11s` 사용
   - 중간 사양 GPU 권장

3. **경량 모델이 필요한 경우**
   - `yolov11n` 또는 `yolov8s` 사용
   - CPU에서도 사용 가능

4. **특수 사용 사례**
   - Transformer 기반 모델이 필요한 경우: `detr-resnet50` 또는 `yolos-small`
   - 안정성이 중요한 경우: `yolov5m`

### 주의사항

1. YOLOv11 모델 사용 시 `ultralytics` 패키지 필요
2. YOLOS/DETR 모델 사용 시 `transformers` 패키지 필요
3. GPU 메모리 사용량은 모델 크기에 따라 다름
4. CPU 사용 시 경량 모델 권장

---

# 번호판 탐지 및 YOLO 라벨링 생성기

이미지에서 번호판을 자동으로 탐지하고 최신 딥러닝 모델을 사용하여 YOLO 형식의 라벨 파일을 생성하는 AI 기반 도구입니다.

## 🚀 주요 기능

- **자동 번호판 탐지**: 다양한 모델(YOLOS, DETR, YOLOv5, YOLOv8)을 사용한 정확한 번호판 탐지
- **YOLO 라벨 생성**: 탐지된 번호판을 YOLO 형식의 라벨로 자동 변환
- **이미지 크기 최적화**: 입력 이미지 크기에 따른 자동 처리 크기 조정
- **시각화 지원**: 탐지 결과를 확인할 수 있는 시각화 이미지 생성
- **일괄 처리**: 디렉토리 내 모든 이미지를 한 번에 처리
- **미탐지 이미지 관리**: 번호판이 탐지되지 않은 이미지를 별도 저장
- **GPU 가속**: 자동 GPU 감지 및 사용으로 빠른 처리
- **유연한 디바이스 제어**: 필요시 CPU 사용 강제 옵션

## 📋 요구사항

### Python 버전
- Python 3.8 이상

### 패키지 관리 옵션

#### 옵션 1: uv 사용 (빠른 설치를 위해 권장)
```bash
# uv가 설치되어 있지 않은 경우 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
# 또는
pip install uv

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies with uv
uv pip install torch torchvision
uv pip install transformers
uv pip install opencv-python
uv pip install Pillow
uv pip install numpy
uv pip install ultralytics
```

Or install from requirements.txt:
```bash
uv pip install -r requirements.txt
```

#### 옵션 2: pip 사용 (전통적인 방법)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install Pillow
pip install numpy
pip install ultralytics
```

Or if you have a requirements.txt file:
```bash
pip install -r requirements.txt
```

### CUDA Support

For GPU acceleration, install the appropriate CUDA version:

#### CUDA 12.1 (Latest)
```bash
uv pip install -r requirements-cuda121.txt
# or
pip install -r requirements-cuda121.txt
```

#### CUDA 12.0
```bash
uv pip install -r requirements-cuda120.txt
# or
pip install -r requirements-cuda120.txt
```

#### CUDA 11.8 (Legacy)
```bash
uv pip install -r requirements-cuda118.txt
# or
pip install -r requirements-cuda118.txt
```

#### CPU-only Installation
```bash
uv pip install -r requirements-cpu.txt
# or
pip install -r requirements-cpu.txt
```

### Verification

After installation, verify your setup:
```bash
# Check PyTorch installation
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check Transformers installation
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Check Ultralytics installation
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"

# Check OpenCV installation
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🛠️ Installation

For detailed installation instructions, please refer to [INSTALL.md](INSTALL.md).

### **Clone Repository**
```bash
git clone <repository-url>
cd <downloaded-git-folder>
```

### Quick Installation

#### Option 1: Using uv (Recommended for fast installation)
```bash
# Create virtual environment with uv
uv venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

#### Using traditional venv
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. **Install Dependencies**

#### With uv (Recommended)
```bash
uv pip install -r requirements.txt
```

#### With pip
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

**Single Image Processing:**
```bash
python src/license_plate_labeler.py -i path/to/image.jpg -o output_directory
```

**Directory Batch Processing:**
```bash
python src/license_plate_labeler.py -i input_directory -o output_directory
```

### Advanced Options

**Adjust Confidence Threshold:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir -c 0.7
```

**Disable Visualization:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --no-viz
```

**Save Undetected Images:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir -e undetected_folder
```

**Set Maximum Processing Size:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 1024
```

**Force CPU Usage:**
```bash
python src/license_plate_labeler.py -i input_dir -o output_dir --force-cpu
```

**Combination of All Options:**
```bash
python src/license_plate_labeler.py \
  -i data/images \
  -o results \
  -c 0.6 \
  --max-size 1024 \
  -e undetected \
  --no-viz \
  --force-cpu
```

## 📝 Command Line Options

| Option | Description |
|--------|-------------|
| `--input` or `-i` | Input image file or directory path (Required) |
| `--output` or `-o` | Output directory path (Required) |
| `--model` | Model to use for detection (default: yolos-small) |
| | Available models: |
| | - `yolos-small`: YOLO + Vision Transformer (90MB) |
| | - `yolos-rego`: YOLOS + 차량+번호판 동시 탐지 (90MB) |
| | - `detr-resnet50`: DETR + ResNet50 (160MB) |
| | - `yolov5m`: YOLOv5 medium (40MB) |
| | - `yolov8s`: YOLOv8 small (22MB) |
| | - `yolov11n`: YOLOv11 nano (5.47MB) |
| | - `yolov11s`: YOLOv11 small (19.2MB) |
| | - `yolov11m`: YOLOv11 medium (40.5MB) |
| | - `yolov11l`: YOLOv11 large (51.2MB) |
| | - `yolov11x`: YOLOv11x (최고 정확도, 114MB) |
| | *(YOLOv11 모델은 처음 사용 시 Hugging Face에서 자동으로 다운로드되어 `~/.cache/license_plate_models` 폴더에 저장됩니다.)*
| `--confidence` or `-c` | Confidence threshold (default 0.5) |
| `--no-viz` | Disable visualization output (default False) |
| `--undetected-dir` or `-e` | Directory to save undetected images (default None) |
| `--max-size` | Maximum processing size (default 800) |
| `--force-cpu` | Force CPU usage (default False) |
| `--token` | Hugging Face API token for model download (use read-only token) |
| `--local-model` | Path to local model file for offline use |
| `--list-models` | List all available models and exit |

## 📁 Output File Structure

After processing, the following files will be generated in the output directory:

```
output_directory/
├── image1.jpg                    # Original image (copied from input)
├── image1.txt                    # YOLO label file (same name as image)
├── image1_detected.jpg           # Detection visualization (optional)
├── image2.png                    # Original image (copied from input)
├── image2.txt                    # YOLO label file (same name as image)
├── image2_detected.jpg           # Detection visualization (optional)
└── ...
```

**Key Features of File Organization:**
- **Original Images**: Source images are copied to the output directory preserving their original filenames and extensions
- **Label Files**: YOLO label files (.txt) have identical filenames to their corresponding images, differing only in extension
- **Visualization Files**: Detection visualization images use the `_detected.jpg` suffix for easy identification
- **Perfect Dataset Structure**: Ready-to-use YOLO training dataset with properly paired images and labels

### YOLO Label File Format

Each `.txt` file follows this format:
```
class_id x_center y_center width height
```

Example:
```
0 0.512500 0.345833 0.125000 0.158333
0 0.687500 0.512500 0.156250 0.145833
```

- `class_id`: Class ID (license plate is 0)
- `x_center, y_center`: Bounding box center point (normalized coordinates, 0.0-1.0)
- `width, height`: Bounding box size (normalized size, 0.0-1.0)

## 🖼️ Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## ⚙️ Performance Optimization

### Image Size Optimization
The program automatically analyzes input image size to determine optimal processing size:

- **Large Images (>800px)**: Downscale to maximum size for improved processing speed
- **Small Images (<400px)**: Upscale to minimum size for better detection accuracy
- **Appropriate Size**: Adjust to multiples of 8 for model efficiency optimization

### GPU Usage
When CUDA is installed, the program automatically detects and uses GPU to improve processing speed. The program will display detailed GPU information including:
- GPU name and device count
- Memory usage statistics
- Automatic fallback to CPU if GPU is unavailable

### Device Selection
- **Automatic GPU Detection**: Uses GPU automatically when CUDA is available
- **Manual CPU Override**: Use `--force-cpu` to disable GPU usage
- **Memory Management**: Automatic GPU memory monitoring and error handling

### Memory Optimization
- Large images are automatically resized to reduce memory usage
- During batch processing, images are processed one by one to prevent memory overflow

## 🔧 Troubleshooting

### Common Errors

**1. Model Loading Failure**
```
모델 로딩 실패: HTTPSConnectionPool...
```
**해결 방법**: 
- 인터넷 연결 확인
- 방화벽 설정 확인
- VPN 연결 해제 시도

**2. GPU Out of Memory**
```
RuntimeError: CUDA out of memory
```
**해결 방법**: 
```bash
# 이미지 처리 크기 줄이기
python src/license_plate_labeler.py -i input_dir -o output_dir --max-size 600

# 또는 CPU 사용 강제
python src/license_plate_labeler.py -i input_dir -o output_dir --force-cpu
```

**3. GPU 미감지**
```
CUDA를 사용할 수 없습니다. CPU를 사용합니다.
```
**해결 방법**: 
- NVIDIA GPU 설치 확인
- CUDA 설치 확인: `nvidia-smi`
- PyTorch CUDA 지원 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- 필요한 경우 CUDA 지원 PyTorch 재설치

## 📞 지원 및 커뮤니티

### 지원 채널
- **GitHub Issues**: 버그 보고 및 기능 요청
- **GitHub Discussions**: 일반 질문 및 토론
- **Wiki**: 추가 문서 및 튜토리얼

### 관련 리소스
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [OpenCV Python 문서](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyTorch Official Documentation](https://pytorch.org/docs/)

## 🙏 감사의 말

### 오픈 소스 프로젝트
- [HuggingFace Transformers](https://huggingface.co/transformers/) - YOLOS 모델 제공
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [Pillow](https://pillow.readthedocs.io/) - 이미지 처리 라이브러리

### 사전 학습된 모델
- [nickmuchi/yolos-small-finetuned-license-plate-detection](https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection) - 번호판 탐지 모델

### 연락처
- **작성자**: lanco.gh@gmail.com
- **GitHub Issues**: 버그 보고 및 기능 요청

### 기여자
이 프로젝트에 기여한 모든 분들께 감사드립니다.

---

**문제가 발생하거나 질문이 있으시면 GitHub Issues를 통해 연락해 주세요!**

**행복한 번호판 탐지 되세요! 🚗📋**

## 🔑 HuggingFace 토큰 설정

### 토큰이 필요한 경우

다음과 같은 경우 HuggingFace 토큰이 필요합니다:
- Private 모델에 접근할 때
- 다운로드 제한이 있는 모델을 사용할 때
- 특정 모델의 최신 버전에 접근할 때

### 토큰 설정 방법

#### 1. 환경 변수로 설정 (권장)

```bash
# Linux/macOS
export HF_TOKEN="your_token_here"

# Windows (Command Prompt)
set HF_TOKEN=your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
```

#### 2. 명령줄 인자로 설정

```bash
python license_plate_labeler.py -i input_dir -o output_dir -t "your_token_here"
# 또는
python license_plate_labeler.py --token "your_token_here" -i input_dir -o output_dir
```

### 토큰 생성 방법

1. [HuggingFace 웹사이트](https://huggingface.co)에 로그인
2. 우측 상단의 프로필 아이콘 클릭
3. Settings 메뉴 선택
4. 왼쪽 사이드바에서 "Access Tokens" 선택
5. "New token" 버튼 클릭
6. 토큰 이름 입력 및 권한 설정
7. "Generate token" 버튼 클릭
8. 생성된 토큰을 안전한 곳에 복사

### 토큰 설정 확인

토큰이 올바르게 설정되었는지 확인하려면:

```bash
# 환경 변수 확인
echo $HF_TOKEN  # Linux/macOS
echo %HF_TOKEN% # Windows Command Prompt
echo $env:HF_TOKEN # Windows PowerShell

# 프로그램 실행 시 토큰 설정 확인
python license_plate_labeler.py -i input_dir -o output_dir
```

### 주의사항

1. 토큰 보안
   - 토큰을 공개 저장소에 커밋하지 마세요
   - 토큰을 다른 사람과 공유하지 마세요
   - 정기적으로 토큰을 갱신하세요

2. 토큰 권한
   - 필요한 최소한의 권한만 부여하세요
   - 읽기 권한만 필요한 경우 write 권한은 부여하지 마세요

3. 토큰 관리
   - 여러 환경에서 사용하는 경우 환경 변수로 설정하는 것을 권장
   - CI/CD 파이프라인에서는 시크릿으로 관리

### 문제 해결

토큰 관련 문제가 발생하면 다음을 확인하세요:

1. 토큰 유효성
   - 토큰이 만료되지 않았는지 확인
   - 토큰이 올바르게 복사되었는지 확인

2. 권한 문제
   - 토큰에 필요한 권한이 부여되어 있는지 확인
   - 모델에 대한 접근 권한이 있는지 확인

3. 환경 변수 문제
   - 환경 변수가 올바른 세션에 설정되어 있는지 확인
   - 대소문자가 정확한지 확인 (HF_TOKEN)

4. 네트워크 문제
   - HuggingFace 서버에 접근 가능한지 확인
   - 프록시 설정이 필요한지 확인

### CI/CD 통합 예시

GitHub Actions에서 토큰 사용 예시:

```yaml
name: License Plate Detection
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: python license_plate_labeler.py -i input_dir -o output_dir
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## 🖥️ CLI 사용 예시

### 기본 사용법

#### Linux/macOS (bash/zsh)
```bash
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolos-small \
 -t <huggingface_token> \
 -c 0.6 \
 --max-size 640 \
 -e ./temp_data/miss_plate
```

#### Windows (Command Prompt)
```cmd
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -m yolos-small ^
 -t <huggingface_token> ^
 -c 0.6 ^
 --max-size 640 ^
 -e .\temp_data\miss_plate
```

#### Windows (PowerShell)
```powershell
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -m yolos-small `
 -t <huggingface_token> `
 -c 0.6 `
 --max-size 640 `
 -e .\temp_data\miss_plate
```

### 다른 사용 예시

#### 1. 단일 이미지 처리

Linux/macOS (bash/zsh):
```bash
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate/image.jpg \
 -o ./temp_data/out_plate \
 -m yolos-small
```

Windows (Command Prompt):
```cmd
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate\image.jpg ^
 -o .\temp_data\out_plate ^
 -m yolos-small
```

Windows (PowerShell):
```powershell
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate\image.jpg `
 -o .\temp_data\out_plate `
 -m yolos-small
```

#### 2. 다른 모델 사용

Linux/macOS (bash/zsh):
```bash
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolov11x \
 -t <huggingface_token>
```

Windows (Command Prompt):
```cmd
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -m yolov11x ^
 -t <huggingface_token>
```

Windows (PowerShell):
```powershell
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -m yolov11x `
 -t <huggingface_token>
```

#### 3. CPU 모드로 실행

Linux/macOS (bash/zsh):
```bash
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolos-small \
 --force-cpu
```

Windows (Command Prompt):
```cmd
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -m yolos-small ^
 --force-cpu
```

Windows (PowerShell):
```powershell
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -m yolos-small `
 --force-cpu
```

#### 4. 시각화 결과 없이 실행

Linux/macOS (bash/zsh):
```bash
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolos-small \
 --no-viz
```

Windows (Command Prompt):
```cmd
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -m yolos-small ^
 --no-viz
```

Windows (PowerShell):
```powershell
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -m yolos-small `
 --no-viz
```

### 주의사항

1. 경로 구분자
   - Linux/macOS: `/` 사용
   - Windows: `\` 사용
   - PowerShell: `\` 또는 `/` 모두 사용 가능

2. 줄 연속 문자
   - Linux/macOS (bash/zsh): `\`
   - Windows Command Prompt: `^`
   - Windows PowerShell: `` ` `` (백틱)

3. 환경 변수 설정
   - Linux/macOS (bash/zsh): `export HF_TOKEN="your_token_here"`
   - Windows Command Prompt: `set HF_TOKEN=your_token_here`
   - Windows PowerShell: `$env:HF_TOKEN="your_token_here"`

4. 디렉토리 구조
   - 입력 디렉토리: 원본 이미지가 있는 디렉토리
   - 출력 디렉토리: 탐지 결과와 라벨 파일이 저장될 디렉토리
   - 미탐지 디렉토리: 번호판이 탐지되지 않은 이미지가 저장될 디렉토리

5. 권한 설정
   - 입력/출력 디렉토리에 대한 읽기/쓰기 권한이 필요합니다
   - 디렉토리가 없는 경우 자동으로 생성됩니다

6. 메모리 사용
   - `--max-size` 파라미터를 조정하여 메모리 사용량을 제어할 수 있습니다
   - GPU 메모리 부족 시 `--force-cpu` 옵션을 사용하세요

7. 성능 최적화
   - 신뢰도 임계값(`-c`)을 조정하여 탐지 정확도를 조절할 수 있습니다
   - 이미지 크기(`--max-size`)를 조정하여 처리 속도를 조절할 수 있습니다

## CLI 실행 방법

### Windows PowerShell

1. 한 줄로 실행:
```powershell
Remove-Item -Recurse -Force .\temp_data\miss_plate\* -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force .\temp_data\out_plate\* -ErrorAction SilentlyContinue; python .\src\license_plate_labeler.py -i .\temp_data\org_plate -o .\temp_data\out_plate -t <your_hugging_face_read_token> -c 0.6 --max-size 640 -e .\temp_data\miss_plate -m yolos-small
```

2. 여러 줄로 실행 (가독성 향상):
```powershell
Remove-Item -Recurse -Force ./temp_data/miss_plate/* -ErrorAction SilentlyContinue; `
Remove-Item -Recurse -Force ./temp_data/out_plate/* -ErrorAction SilentlyContinue; `
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -t <your_hugging_face_read_token> `
 -c 0.6 `
 --max-size 640 `
 -e .\temp_data\miss_plate `
 -m yolos-small
```

### Linux

```bash
rm -rf ./temp_data/miss_plate/* && rm -rf ./temp_data/out_plate/* && python ./src/license_plate_labeler.py -i ./temp_data/org_plate -o ./temp_data/out_plate -t <your_hugging_face_read_token> -c 0.25 --max-size 1024 -e ./temp_data/miss_plate -m yolov5s
```

### 매개변수 설명

- `-i`: 입력 이미지 디렉토리
- `-o`: 출력 이미지 디렉토리
- `-t`: Hugging Face 읽기 토큰
- `-c`: 신뢰도 임계값 (0.0 ~ 1.0)
- `--max-size`: 최대 이미지 크기
- `-e`: 미감지 이미지 저장 디렉토리
- `-m`: 사용할 모델 (예: yolos-small, yolov5s)

### 주의사항

1. Hugging Face 토큰은 `<your_hugging_face_read_token>`을 실제 토큰으로 교체해야 합니다.
2. 디렉토리 경로는 실제 환경에 맞게 조정해야 합니다.
3. Windows에서는 PowerShell을 사용하는 것이 권장됩니다.
4. Linux에서는 bash 쉘을 사용합니다.

## 모델 비교표

| Model | Framework | Size | Accuracy (mAP@50) | Speed | GPU Memory | CPU Usage |
|-------|-----------|------|-------------------|-------|------------|-----------|
| yolov11x | YOLOv11 | 114MB | 0.9813 | ⭐ | 8GB+ | ❌ |
| yolov11l | YOLOv11 | 51.2MB | 0.9789 | ⭐⭐ | 6GB+ | ❌ |
| yolov11m | YOLOv11 | 40.5MB | 0.9765 | ⭐⭐⭐ | 4GB+ | ❌ |
| detr-resnet-50 | DETR | 159MB | 0.9756 | ⭐⭐ | 6GB+ | ❌ |
| yolov11s | YOLOv11 | 19.2MB | 0.9742 | ⭐⭐⭐⭐ | 2GB+ | ⚠️ |
| yolos-small-finetuned | YOLOS | 90MB | 0.9731 | ⭐⭐⭐ | 2GB+ | ⚠️ |
| yolos-rego | YOLOS | 90MB | 0.9728 | ⭐⭐⭐ | 2GB+ | ⚠️ |
| yolov11n | YOLOv11 | 5.47MB | 0.9715 | ⭐⭐⭐⭐⭐ | 1GB+ | ✅ |
| yolov5m | YOLOv5 | 40MB | 0.9702 | ⭐⭐⭐⭐⭐ | 1GB+ | ✅ |
| yolov8-lp-mkgoud | YOLOv8 | 6.24MB | 0.9698 | ⭐⭐⭐⭐ | 1GB+ | ✅ |

### 범례
- ⭐: Speed (⭐: Slow, ⭐⭐⭐⭐⭐: Fast)
- ✅: Recommended
- ⚠️: Possible but slow
- ❌: Not recommended

### 모델 선택 가이드

1. **최고 정확도가 필요한 경우**
   - `yolov11x` 사용
   - 고사양 GPU 필요

2. **균형잡힌 성능이 필요한 경우**
   - `yolov11m` 또는 `yolov11s` 사용
   - 중간 사양 GPU 권장

3. **경량 모델이 필요한 경우**
   - `yolov11n` 또는 `yolov8s` 사용
   - CPU에서도 사용 가능

4. **특수 사용 사례**
   - Transformer 기반 모델이 필요한 경우: `detr-resnet50` 또는 `yolos-small`
   - 안정성이 중요한 경우: `yolov5m`

### 주의사항

1. YOLOv11 모델 사용 시 `ultralytics` 패키지 필요
2. YOLOS/DETR 모델 사용 시 `transformers` 패키지 필요
3. GPU 메모리 사용량은 모델 크기에 따라 다름
4. CPU 사용 시 경량 모델 권장