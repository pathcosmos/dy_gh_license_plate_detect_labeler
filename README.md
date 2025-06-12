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

## 🛠️ Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd <downloaded-git-folder>
```

2. **Create Virtual Environment (Recommended)**

#### Using uv (Faster)
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

| Option | Short Form | Description | Default |
|--------|------------|-------------|---------|
| `--input` | `-i` | Input image file or directory path | Required |
| `--output` | `-o` | Output directory path | Required |
| `--confidence` | `-c` | Confidence threshold (0.0-1.0) | 0.5 |
| `--no-viz` | - | Disable visualization output | False |
| `--undetected-dir` | `-e` | Directory to save undetected images | None |
| `--max-size` | - | Maximum processing size (longest edge) | 800 |
| `--force-cpu` | - | Force CPU usage (disable GPU) | False |

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
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   
   # Or on Windows
   set HUGGINGFACE_HUB_TOKEN=your_token_here
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

### Recommended Models
- **yolos-small**: YOLO + Vision Transformer, fine-tuned for license plate detection (90MB)
- **detr-resnet50**: DETR + ResNet50 backbone, specialized for license plate detection (160MB)
- **yolov5m**: YOLOv5 medium model, specialized for license plate detection (40MB)

### Model Categories
1. **🔥 Recommended License Plate Models**
   - yolos-small
   - detr-resnet50

2. **🏆 YOLOS-based (Transformer)**
   - yolos-small
   - yolos-rego
   - yolos-base

3. **🎯 DETR-based (Detection Transformer)**
   - detr-resnet50
   - detr-resnet-50

4. **⚡ YOLOv8-based**
   - yolov8s

5. **🔧 YOLOv5-based**
   - yolov5m

### Performance Comparison

#### Speed Ranking (Fastest to Slowest)
1. yolov5m (YOLOv5)
2. yolos-small (YOLOS)
3. detr-resnet50 (DETR)

#### Accuracy Ranking (Highest to Lowest)
1. detr-resnet50 (DETR)
2. yolos-small (YOLOS)
3. yolov5m (YOLOv5)

#### Model Size Ranking (Smallest to Largest)
1. yolov5m (40MB)
2. yolos-small (90MB)
3. detr-resnet50 (160MB)

#### Recommended Use Cases
- Real-time Processing: yolov5m
- Highest Accuracy: detr-resnet50
- Balanced Performance: yolos-small
- Stability Priority: yolos-small

---
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

# uv로 의존성 설치
uv pip install torch torchvision
uv pip install transformers
uv pip install opencv-python
uv pip install Pillow
uv pip install numpy
uv pip install ultralytics
```

또는 requirements.txt에서 설치:
```bash
uv pip install -r requirements.txt
```

#### 옵션 2: pip 사용 (전통적인 방법)
```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install Pillow
pip install numpy
pip install ultralytics
```

또는 requirements.txt 파일이 있는 경우:
```bash
pip install -r requirements.txt
```

## 📊 사용 가능한 모델

### 추천 모델
- **yolos-small**: YOLO + Vision Transformer, 번호판 탐지용 파인튜닝 (90MB)
- **detr-resnet50**: DETR + ResNet50 백본, 번호판 탐지 특화 (160MB)
- **yolov5m**: YOLOv5 medium 모델, 번호판 탐지 특화 (40MB)

### 모델 카테고리
1. **🔥 추천 번호판 모델**
   - yolos-small
   - detr-resnet50

2. **🏆 YOLOS 기반 (Transformer)**
   - yolos-small
   - yolos-rego
   - yolos-base

3. **🎯 DETR 기반 (Detection Transformer)**
   - detr-resnet50
   - detr-resnet-50

4. **⚡ YOLOv8 기반**
   - yolov8s

5. **🔧 YOLOv5 기반**
   - yolov5m

### 성능 비교

#### 속도 순위 (빠른 순)
1. yolov5m (YOLOv5)
2. yolos-small (YOLOS)
3. detr-resnet50 (DETR)

#### 정확도 순위 (높은 순)
1. detr-resnet50 (DETR)
2. yolos-small (YOLOS)
3. yolov5m (YOLOv5)

#### 모델 크기 순위 (작은 순)
1. yolov5m (40MB)
2. yolos-small (90MB)
3. detr-resnet50 (160MB)

#### 추천 사용 사례
- 실시간 처리: yolov5m
- 최고 정확도: detr-resnet50
- 균형잡힌 성능: yolos-small
- 안정성 우선: yolos-small

## 📈 성능 지표

### 테스트 환경
- **CPU**: Intel i5-13500
- **GPU**: NVIDIA RTX 4070
- **RAM**: 32GB DDR5
- **이미지**: 1600x1200 JPEG

### 처리 속도
- **CPU 처리**: 이미지당 ~2-3초
- **GPU 처리**: 이미지당 ~0.5-1초 (GPU에 따라 다름)
- **일괄 처리**: 시간당 ~1000장 (GPU), ~300장 (CPU)

### GPU 요구사항
- **최소 VRAM**: 4GB 권장
- **최적 VRAM**: 대용량 이미지의 경우 8GB 이상
- **CUDA 버전**: PyTorch 설치와 호환되는 버전

## 📝 사용 예시

### 기본 사용법
```bash
# 단일 이미지 처리
python src/license_plate_labeler.py -i 이미지.jpg -o 출력_디렉토리

# 디렉토리 일괄 처리
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리
```

### 고급 옵션
```bash
# 신뢰도 임계값 조정
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 -c 0.7

# 시각화 비활성화
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 --no-viz

# 미탐지 이미지 저장
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 -e 미탐지_폴더

# 최대 처리 크기 설정
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 --max-size 1024

# CPU 강제 사용
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 --force-cpu
```

## 🔧 문제 해결

### 일반적인 오류

1. **모델 로딩 실패**
```
모델 로딩 실패: HTTPSConnectionPool...
```
**해결 방법**: 
- 인터넷 연결 확인
- 방화벽 설정 확인
- VPN 연결 해제 시도

2. **GPU 메모리 부족**
```
RuntimeError: CUDA out of memory
```
**해결 방법**: 
```bash
# 이미지 처리 크기 줄이기
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 --max-size 600

# 또는 CPU 사용 강제
python src/license_plate_labeler.py -i 입력_디렉토리 -o 출력_디렉토리 --force-cpu
```

3. **GPU 미감지**
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
- [PyTorch 공식 문서](https://pytorch.org/docs/)

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
