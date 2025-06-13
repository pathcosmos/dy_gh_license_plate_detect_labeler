# License Plate Detection System

## 🚀 Overview
This system provides a comprehensive solution for license plate detection using various YOLO-based models. It supports multiple model architectures and sizes to cater to different use cases and hardware requirements.

## 📋 Features
- Multiple model support (YOLOv8, YOLOv5)
- Direct model download capability
- Configurable confidence thresholds
- Support for both single images and directories
- Visualization options
- Undetected image tracking
- Comprehensive logging

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Ultralytics
- OpenCV
- Pillow
- Hugging Face Hub

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face token:
```bash
export HF_TOKEN=your_hugging_face_token
```

## 🔑 HuggingFace Token Setup
### When Token is Required
- Accessing private models
- Using models with download restrictions
- Accessing latest versions

### Setting Up Token
1. **Environment Variable (Recommended)**
   ```bash
   # Linux/macOS
   export HF_TOKEN="your_token_here"
   
   # Windows (Command Prompt)
   set HF_TOKEN=your_token_here
   
   # Windows (PowerShell)
   $env:HF_TOKEN="your_token_here"
   ```

2. **Command Line Argument**
   ```bash
   python license_plate_labeler.py -t your_token_here
   ```

### Getting a Token
1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
2. Create new token
3. Copy token value

## 🖥️ CLI Usage Examples

### Basic Usage

#### Linux (bash)
```bash
# 방법 1: 토큰을 직접 명령어에 포함
rm -f ./temp_data/miss_plate/* && rm -f ./temp_data/out_plate/* &&
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolos-rego \
 -t <HF_TOKEN> \
 -c 0.6 \
 --max-size 640 \
 -e ./temp_data/miss_plate

# 방법 2: 환경 변수로 토큰 설정
export HF_TOKEN=<HF_TOKEN>
rm -f ./temp_data/miss_plate/* && rm -f ./temp_data/out_plate/* &&
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -c 0.6 \
 --max-size 640 \
 -e ./temp_data/miss_plate \
 -m yolos-small
```

#### macOS (zsh)
```bash
# 방법 1: 토큰을 직접 명령어에 포함
rm -f ./temp_data/miss_plate/* && rm -f ./temp_data/out_plate/* &&
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -m yolos-rego \
 -t <HF_TOKEN> \
 -c 0.6 \
 --max-size 640 \
 -e ./temp_data/miss_plate

# 방법 2: 환경 변수로 토큰 설정
export HF_TOKEN=<HF_TOKEN>
rm -f ./temp_data/miss_plate/* && rm -f ./temp_data/out_plate/* &&
python ./src/license_plate_labeler.py \
 -i ./temp_data/org_plate \
 -o ./temp_data/out_plate \
 -c 0.6 \
 --max-size 640 \
 -e ./temp_data/miss_plate \
 -m yolos-small
```

#### Windows (Command Prompt)
```batch
:: 방법 1: 토큰을 직접 명령어에 포함
del /Q .\temp_data\miss_plate\* && del /Q .\temp_data\out_plate\* &&
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -m yolos-rego ^
 -t <HF_TOKEN> ^
 -c 0.6 ^
 --max-size 640 ^
 -e .\temp_data\miss_plate

:: 방법 2: 환경 변수로 토큰 설정
set HF_TOKEN=<HF_TOKEN>
del /Q .\temp_data\miss_plate\* && del /Q .\temp_data\out_plate\* &&
python .\src\license_plate_labeler.py ^
 -i .\temp_data\org_plate ^
 -o .\temp_data\out_plate ^
 -c 0.6 ^
 --max-size 640 ^
 -e .\temp_data\miss_plate ^
 -m yolos-small
```

#### Windows (PowerShell)
```powershell
# 방법 1: 토큰을 직접 명령어에 포함
Remove-Item -Path .\temp_data\miss_plate\* -Force; Remove-Item -Path .\temp_data\out_plate\* -Force;
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -m yolos-rego `
 -t <HF_TOKEN> `
 -c 0.6 `
 --max-size 640 `
 -e .\temp_data\miss_plate

# 방법 2: 환경 변수로 토큰 설정
$env:HF_TOKEN="<HF_TOKEN>"
Remove-Item -Path .\temp_data\miss_plate\* -Force; Remove-Item -Path .\temp_data\out_plate\* -Force;
python .\src\license_plate_labeler.py `
 -i .\temp_data\org_plate `
 -o .\temp_data\out_plate `
 -c 0.6 `
 --max-size 640 `
 -e .\temp_data\miss_plate `
 -m yolos-small
```

### 명령어 설명
- `rm -f` (Linux/macOS) 또는 `del /Q` (Windows CMD) 또는 `Remove-Item -Force` (PowerShell): 이전 실행 결과 파일 삭제
- `-i`: 입력 이미지 또는 디렉토리 경로
- `-o`: 결과를 저장할 출력 디렉토리 경로
- `-m`: 사용할 모델 선택
- `-t`: HuggingFace 토큰 (선택사항)
- `-c`: 신뢰도 임계값 (0.0-1.0, 기본값: 0.5)
- `--max-size`: 처리할 최대 이미지 크기 (기본값: 800)
- `-e`: 미감지된 이미지를 저장할 디렉토리 (선택사항)

### 주의사항
1. **경로 구분자**
   - Windows: `\` 또는 `/` 사용 가능
   - Linux/macOS: `/` 사용

2. **줄 연결 문자**
   - Windows CMD: `^`
   - Windows PowerShell: `` ` ``
   - Linux/macOS: `\`

3. **환경 변수 설정**
   - Linux/macOS: `export HF_TOKEN=<token>`
   - Windows CMD: `set HF_TOKEN=<token>`
   - Windows PowerShell: `$env:HF_TOKEN="<token>"`

4. **디렉토리 구조**
   - 입력 디렉토리가 존재해야 함
   - 출력 디렉토리는 자동 생성됨
   - 미감지 디렉토리는 지정 시 자동 생성됨

5. **권한 설정**
   - 입력 디렉토리에 대한 읽기 권한 필요
   - 출력 디렉토리에 대한 쓰기 권한 필요
   - 미감지 디렉토리에 대한 쓰기 권한 필요 (지정 시)

### Additional Examples
```bash
# Process single image
python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir

# Use different model
python license_plate_labeler.py -i input.jpg -o output -m yolov8m-lp-mkgoud

# Run in CPU mode
python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir --device cpu

# Run without visualization
python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir --no-vis
```

## 📊 Available Models

### YOLOS 기반 모델 (Vision Transformer)
- `yolos-small`: YOLO + Vision Transformer, 번호판 전용 파인튜닝 (90MB)
- `yolos-rego`: 차량+번호판 동시 탐지, 735 이미지로 200 에포크 훈련 (90MB)

### DETR 기반 모델 (Detection Transformer)
- `detr-resnet50`: DETR + ResNet50 백본, 번호판 탐지 전용 (160MB)

### YOLOv5 기반 모델
- `yolov5m`: YOLOv5 medium 모델, 번호판 탐지 특화 (40MB)
- `yolov5n-lp`: YOLOv5 nano 모델, 번호판 탐지 특화 (3.8MB)
- `yolov5s-lp`: YOLOv5 small 모델, 번호판 탐지 특화 (14MB)
- `yolov5m-lp`: YOLOv5 medium 모델, 번호판 탐지 특화 (40MB)

### YOLOv8 기반 모델
- `yolov8s`: 기본 YOLOv8 small 모델 (22MB) - 범용 객체 탐지
- `yolov8m-lp-mkgoud`: YOLOv8 medium 모델, 번호판 탐지 특화 (43MB)
- `yolov8m-lp-koushim`: YOLOv8 medium 모델, 번호판 탐지 특화 (43MB)
- `yolov8-lp-yasir`: YOLOv8 기반 번호판 탐지 모델 (6.24MB)

### YOLOv11 기반 모델
- `yolov11n`: YOLOv11 nano 모델, 가장 작은 크기 (5.47MB)
- `yolov11s`: YOLOv11 small 모델, 균형잡힌 성능 (19.2MB)
- `yolov11m`: YOLOv11 medium 모델, 높은 정확도 (40.5MB)
- `yolov11l`: YOLOv11 large 모델, 매우 높은 정확도 (51.2MB)
- `yolov11x`: YOLOv11x 모델, 최고 정확도 (114MB)

### 기본 모델 (호환성 유지)
- `yolos-base`: YOLOS Base model (범용 객체 탐지)
- `detr-resnet-50`: DETR ResNet-50 model (범용 객체 탐지)

## 🧪 Testing
### Test Environment Setup
1. Set HuggingFace token:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. Install test dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests
```bash
# Run all tests
./tests/run_tests.sh

# Run specific test
python -m unittest tests/test_license_plate_labeler.py -k test_yolov8_models
```

### Test Cases
- Model loading and initialization
- Image processing with different models
- Confidence threshold variations
- Maximum size parameter testing
- Error handling and edge cases

### Test Results
- Success/failure tracking for each model
- Detailed error reporting
- Performance metrics
- Memory usage monitoring

## ⚠️ Important Notes
1. **Path Separators**
   - Windows: Use `\` or `/`
   - Linux/macOS: Use `/`

2. **Line Continuation**
   - Windows CMD: `^`
   - Windows PowerShell: `` ` ``
   - Linux/macOS: `\`

3. **Directory Structure**
   - Input directory must exist
   - Output directory will be created if not exists
   - Undetected directory will be created if specified

4. **Permissions**
   - Read access to input directory
   - Write access to output directory
   - Write access to undetected directory (if specified)

5. **Memory Usage**
   - Larger models require more GPU memory
   - Consider using smaller models for limited resources
   - Monitor memory usage during processing

6. **Performance Optimization**
   - Use appropriate model size for your needs
   - Adjust confidence threshold based on requirements
   - Consider using CPU mode for smaller models
   - Use max-size parameter to control memory usage

## 🔍 Troubleshooting
1. **Model Download Issues**
   - Verify internet connection
   - Check HuggingFace token
   - Ensure sufficient disk space
   - Check model availability

2. **Processing Errors**
   - Verify input image format
   - Check file permissions
   - Monitor memory usage
   - Review error logs

3. **Performance Issues**
   - Try smaller model
   - Adjust confidence threshold
   - Reduce max-size parameter
   - Use CPU mode if GPU memory is limited

## 🔄 CI/CD Integration
Example GitHub Actions workflow:
```yaml
name: License Plate Detection Tests

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
      - name: Set HuggingFace token
        run: echo "HF_TOKEN=${{ secrets.HF_TOKEN }}" >> $GITHUB_ENV
      - name: Run tests
        run: ./tests/run_tests.sh
```

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.