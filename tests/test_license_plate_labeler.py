import os
import sys
import unittest
import shutil
from pathlib import Path
import subprocess
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from license_plate_labeler import LicensePlateYOLOLabeler

class TestLicensePlateLabeler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 실행"""
        # 테스트 디렉토리 설정
        cls.test_dir = Path(__file__).parent
        cls.temp_dir = cls.test_dir / "temp_test"
        cls.input_dir = cls.temp_dir / "input"
        cls.output_dir = cls.temp_dir / "output"
        cls.miss_dir = cls.temp_dir / "miss"
        
        # 테스트 디렉토리 생성
        for dir_path in [cls.input_dir, cls.output_dir, cls.miss_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 테스트 이미지 복사
        test_data_dir = Path(__file__).parent.parent / "temp_data" / "org_plate"
        if test_data_dir.exists():
            for img_file in test_data_dir.glob("*.jpg"):
                shutil.copy2(img_file, cls.input_dir)
    
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 실행"""
        # 테스트 디렉토리 정리
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """각 테스트 시작 전 실행"""
        # 출력 디렉토리 초기화
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 미탐지 디렉토리 초기화
        if self.miss_dir.exists():
            shutil.rmtree(self.miss_dir)
        self.miss_dir.mkdir(parents=True, exist_ok=True)
    
    def test_yolos_rego_model(self):
        """YOLOS-REGO 모델 테스트"""
        # 테스트 실행
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "src" / "license_plate_labeler.py"),
            "-i", str(self.input_dir),
            "-o", str(self.output_dir),
            "-m", "yolos-rego",
            "-c", "0.6",
            "--max-size", "1600",
            "-e", str(self.miss_dir)
        ]
        
        # Hugging Face 토큰이 환경변수에 있는 경우 추가
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            cmd.extend(["-t", hf_token])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 테스트 결과 검증
        self.assertEqual(result.returncode, 0, f"명령 실행 실패: {result.stderr}")
        
        # 출력 파일 확인
        output_files = list(self.output_dir.glob("*.jpg"))
        self.assertGreater(len(output_files), 0, "출력 이미지가 없습니다.")
        
        # 라벨 파일 확인
        label_files = list(self.output_dir.glob("*.txt"))
        self.assertGreater(len(label_files), 0, "라벨 파일이 없습니다.")
        
    def test_default_model(self):
        """DEFAULT 모델 테스트"""
        # 테스트 실행
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "src" / "license_plate_labeler.py"),
            "-i", str(self.input_dir),
            "-o", str(self.output_dir),
            "-c", "0.6",
            "--max-size", "640",
            "-e", str(self.miss_dir)
        ]
        
        # Hugging Face 토큰이 환경변수에 있는 경우 추가
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            cmd.extend(["-t", hf_token])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 테스트 결과 검증
        self.assertEqual(result.returncode, 0, f"명령 실행 실패: {result.stderr}")
        
        # 출력 파일 확인
        output_files = list(self.output_dir.glob("*.jpg"))
        self.assertGreater(len(output_files), 0, "출력 이미지가 없습니다.")
        
        # 라벨 파일 확인
        label_files = list(self.output_dir.glob("*.txt"))
        self.assertGreater(len(label_files), 0, "라벨 파일이 없습니다.")
    
    def test_yolov11_models(self):
        """YOLOv11 모델들 테스트"""
        models = ["yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"]
        
        for model in models:
            with self.subTest(model=model):
                # 테스트 실행
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent.parent / "src" / "license_plate_labeler.py"),
                    "-i", str(self.input_dir),
                    "-o", str(self.output_dir),
                    "-m", model,
                    "-c", "0.6",
                    "--max-size", "640",
                    "-e", str(self.miss_dir)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 테스트 결과 검증
                self.assertEqual(result.returncode, 0, f"{model} 모델 실행 실패: {result.stderr}")
                
                # 출력 파일 확인
                output_files = list(self.output_dir.glob("*.jpg"))
                self.assertGreater(len(output_files), 0, f"{model} 모델: 출력 이미지가 없습니다.")
                
                # 라벨 파일 확인
                label_files = list(self.output_dir.glob("*.txt"))
                self.assertGreater(len(label_files), 0, f"{model} 모델: 라벨 파일이 없습니다.")
    
    def test_different_confidence_thresholds(self):
        """다양한 신뢰도 임계값 테스트"""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                # 테스트 실행
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent.parent / "src" / "license_plate_labeler.py"),
                    "-i", str(self.input_dir),
                    "-o", str(self.output_dir),
                    "-m", "yolos-small",
                    "-c", str(threshold),
                    "--max-size", "640",
                    "-e", str(self.miss_dir)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 테스트 결과 검증
                self.assertEqual(result.returncode, 0, f"임계값 {threshold} 실행 실패: {result.stderr}")
                
                # 출력 파일 확인
                output_files = list(self.output_dir.glob("*.jpg"))
                self.assertGreater(len(output_files), 0, f"임계값 {threshold}: 출력 이미지가 없습니다.")
    
    def test_different_max_sizes(self):
        """다양한 최대 크기 테스트"""
        sizes = [800, 1200, 1600, 2000]
        
        for size in sizes:
            with self.subTest(size=size):
                # 테스트 실행
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent.parent / "src" / "license_plate_labeler.py"),
                    "-i", str(self.input_dir),
                    "-o", str(self.output_dir),
                    "-m", "yolos-small",
                    "-c", "0.6",
                    "--max-size", str(size),
                    "-e", str(self.miss_dir)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 테스트 결과 검증
                self.assertEqual(result.returncode, 0, f"크기 {size} 실행 실패: {result.stderr}")
                
                # 출력 파일 확인
                output_files = list(self.output_dir.glob("*.jpg"))
                self.assertGreater(len(output_files), 0, f"크기 {size}: 출력 이미지가 없습니다.")

if __name__ == '__main__':
    unittest.main() 