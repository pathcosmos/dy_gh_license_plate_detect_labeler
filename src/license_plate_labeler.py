import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import shutil
import urllib.request
import ssl
import requests
from urllib.parse import urlparse
from PIL import Image
from transformers import (
    YolosImageProcessor, 
    YolosForObjectDetection,
    DetrImageProcessor, 
    DetrForObjectDetection,
    AutoModelForObjectDetection,
    AutoImageProcessor
)
from ultralytics import YOLO
from typing import Any, List, Dict, Tuple
import logging
import traceback
import time
import sys
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LicensePlateYOLOLabeler:
    """YOLO 기반 번호판 탐지 라벨러"""
    
    AVAILABLE_MODELS = {
        # YOLOv11 기반 모델들 (최고 성능)
        "yolov11x": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11x 모델, 최고 정확도 (114MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1x.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "114MB",
            "performance": "최고",
            "metrics": {
                "precision": 0.9893,
                "recall": 0.9508,
                "mAP@50": 0.9813,
                "mAP@50-95": 0.7260
            },
            "pros": ["최고 정확도", "강력한 특징 추출", "복잡한 케이스 처리 우수"],
            "cons": ["매우 큰 모델 크기", "높은 GPU 메모리 요구사항"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1x.pt"
        },
        "yolov11l": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 large 모델, 매우 높은 정확도 (51.2MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1l.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "51.2MB",
            "performance": "매우 높음",
            "metrics": {
                "precision": 0.985,
                "recall": 0.948,
                "mAP@50": 0.978,
                "mAP@50-95": 0.72
            },
            "pros": ["매우 높은 정확도", "강력한 특징 추출", "실시간 처리 가능"],
            "cons": ["큰 모델 크기"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1l.pt"
        },
        "yolov11m": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 medium 모델, 높은 정확도 (40.5MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1m.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "40.5MB",
            "performance": "높음",
            "metrics": {
                "precision": 0.98,
                "recall": 0.95,
                "mAP@50": 0.97,
                "mAP@50-95": 0.70
            },
            "pros": ["높은 정확도", "적절한 추론 속도", "실시간 처리 가능"],
            "cons": ["x 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1m.pt"
        },
        "yolov11s": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 small 모델, 균형잡힌 성능 (19.2MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "19.2MB",
            "performance": "중간",
            "metrics": {
                "precision": 0.97,
                "recall": 0.94,
                "mAP@50": 0.96,
                "mAP@50-95": 0.68
            },
            "pros": ["균형잡힌 성능", "적절한 모델 크기", "실시간 처리 가능"],
            "cons": ["x 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1s.pt"
        },
        "yolov11n": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 nano 모델, 가장 작은 크기 (5.47MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "5.47MB",
            "performance": "중간",
            "metrics": {
                "precision": 0.95,
                "recall": 0.92,
                "mAP@50": 0.94,
                "mAP@50-95": 0.65
            },
            "pros": ["매우 작은 모델 크기", "빠른 추론", "저사양 기기 적합"],
            "cons": ["다른 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1n.pt"
        },
        
        # YOLOS 모델
        "yolos-small": {
            "name": "hustvl/yolos-small",
            "description": "YOLOS small 모델, 경량화된 버전 (14MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-small/resolve/main/pytorch_model.bin",
            "processor_type": "YOLOS",
            "framework": "transformers",
            "size": "14MB",
            "performance": "낮음",
            "verified": True
        },
        "yolos-base": {
            "name": "hustvl/yolos-base",
            "description": "YOLOS base 모델, 균형잡힌 성능 (24MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-base/resolve/main/pytorch_model.bin",
            "processor_type": "YOLOS",
            "framework": "transformers",
            "size": "24MB",
            "performance": "중간",
            "verified": True
        },
        "yolos-tiny": {
            "name": "hustvl/yolos-tiny",
            "description": "YOLOS tiny 모델, 초경량 버전 (6MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-tiny/resolve/main/pytorch_model.bin",
            "processor_type": "YOLOS",
            "framework": "transformers",
            "size": "6MB",
            "performance": "낮음",
            "verified": True
        },
        
        # DETR 모델
        "detr-resnet-50": {
            "name": "facebook/detr-resnet-50",
            "description": "DETR ResNet-50 모델, 기본 버전 (159MB)",
            "download_uri": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/pytorch_model.bin",
            "processor_type": "DETR",
            "framework": "transformers",
            "size": "159MB",
            "performance": "높음",
            "verified": True
        },
        "detr-resnet-101": {
            "name": "facebook/detr-resnet-101",
            "description": "DETR ResNet-101 모델, 고성능 버전 (232MB)",
            "download_uri": "https://huggingface.co/facebook/detr-resnet-101/resolve/main/pytorch_model.bin",
            "processor_type": "DETR",
            "framework": "transformers",
            "size": "232MB",
            "performance": "매우 높음",
            "verified": True
        },
        
        
        
        # YOLOv8 모델
        "yolov8-lp-yasir": {
            "name": "yasirfaizahmed/license-plate-object-detection",
            "description": "YOLOv8 nano 번호판 탐지 모델 (yasirfaizahmed, 6.24MB)",
            "download_uri": "https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt",
            "processor_type": "YOLOv8",
            "framework": "ultralytics",
            "size": "6.24MB",
            "performance": "빠름",
            "metrics": {
                "precision": 0.92,
                "recall": 0.89,
                "mAP@50": 0.91
            },
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "Apache-2.0",
            "model_file": "best.pt"
        },
        "yolov8-lp-koushim": {
            "name": "Koushim/yolov8-license-plate-detection",
            "description": "YOLOv8 nano 번호판 탐지 모델 (Koushim, 6.25MB)",
            "download_uri": "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
            "processor_type": "YOLOv8",
            "framework": "ultralytics",
            "size": "6.25MB",
            "performance": "빠름",
            "metrics": {
                "precision": 0.93,
                "recall": 0.90,
                "mAP@50": 0.92
            },
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt"
        },
        # YOLOv11 모델
        
        "yolov8-lp-mkgoud": {
            "name": "MKgoud/License-Plate-Recognizer",
            "description": "YOLOv8 번호판 탐지 모델 (MKgoud, 6.24MB)",
            "download_uri": "https://huggingface.co/MKgoud/License-Plate-Recognizer/resolve/main/LP-detection.pt",
            "processor_type": "YOLOv8",
            "framework": "ultralytics",
            "size": "6.24MB",
            "performance": "빠름",
            "metrics": {
                "precision": 0.94,
                "recall": 0.91,
                "mAP@50": 0.93
            },
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "LP-detection.pt"
        },
        # YOLOv5 모델
        "yolov5n": {
            "name": "keremberke/yolov5n-license-plate",
            "description": "YOLOv5 nano 모델, 초경량 버전 (7MB)",
            "download_uri": "https://huggingface.co/keremberke/yolov5n-license-plate/resolve/main/best.pt",
            "processor_type": "YOLOv5",
            "framework": "ultralytics",
            "size": "7MB",
            "performance": "낮음",
            "metrics": {
                "mAP@0.5": 0.978
            },
            "pros": ["매우 작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt"
        },
        "yolov5s": {
            "name": "keremberke/yolov5s-license-plate",
            "description": "YOLOv5 small 모델, 경량화된 버전 (14MB)",
            "download_uri": "https://huggingface.co/keremberke/yolov5s-license-plate/resolve/main/best.pt",
            "processor_type": "YOLOv5",
            "framework": "ultralytics",
            "size": "14MB",
            "performance": "중간",
            "metrics": {
                "mAP@0.5": 0.985
            },
            "pros": ["적절한 크기", "균형잡힌 성능", "실시간 처리 가능"],
            "cons": ["nano 버전보다 느림"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt"
        },
        "yolov5m": {
            "name": "keremberke/yolov5m-license-plate",
            "description": "YOLOv5 medium 모델, 번호판 탐지 특화 (40MB)",
            "download_uri": "https://huggingface.co/keremberke/yolov5m-license-plate/resolve/main/best.pt",
            "processor_type": "YOLOv5",
            "framework": "ultralytics",
            "size": "40MB",
            "performance": "높음",
            "metrics": {
                "mAP@0.5": 0.988
            },
            "pros": ["높은 정확도", "안정적인 탐지", "복잡한 케이스 처리 우수"],
            "cons": ["큰 모델 크기", "추론 속도 느림"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt"
        },
    }
    
    # 성능 비교 정보 업데이트
    PERFORMANCE_COMPARISON = {
        "속도_순위": [
            "yolov5m (YOLOv5)",
            "yolos-small (YOLOS)",
            "detr-resnet50 (DETR)"
        ],
        "정확도_순위": [
            "detr-resnet50 (DETR)",
            "yolos-small (YOLOS)",
            "yolov5m (YOLOv5)"
        ],
        "모델크기_순위": [
            "yolov5m (40MB)",
            "yolos-small (90MB)",
            "detr-resnet50 (160MB)"
        ],
        "추천_용도": {
            "실시간_처리": "yolov5m",
            "최고_정확도": "detr-resnet50", 
            "균형잡힌_성능": "yolos-small",
            "안정성_우선": "yolos-small"
        }
    }

    def list_available_models(self):
        """사용 가능한 모델 목록을 출력합니다."""
        print("\n=== 사용 가능한 모델 목록 ===")
        
        # 모델 카테고리별로 그룹화
        categories = {
            "YOLOv8 모델": ["yolov8-lp-yasir", "yolov8m-lp-mkgoud", "yolov8m-lp-koushim"],
            "YOLOv5 모델": ["yolov5n-lp", "yolov5s-lp", "yolov5m-lp", 
                        "yolov5n-anpr", "yolov5s-anpr", "yolov5m-anpr"]
        }
        
        for category, models in categories.items():
            print(f"\n[{category}]")
            for model_id in models:
                if model_id in self.AVAILABLE_MODELS:
                    model = self.AVAILABLE_MODELS[model_id]
                    print(f"\n{model_id}:")
                    print(f"  설명: {model['description']}")
                    print(f"  크기: {model['size']}")
                    print(f"  성능: {model['performance']}")
                    print(f"  장점: {', '.join(model['pros'])}")
                    print(f"  단점: {', '.join(model['cons'])}")
                    print(f"  라이선스: {model['license']}")
                    if model.get('direct_download'):
                        print("  다운로드: 직접 다운로드 가능")
                    print(f"  사용 예시: python license_plate_labeler.py -i input.jpg -o output -m {model_id}")

        print("\n=== 모델 선택 가이드 ===")
        print("1. 실시간 처리가 필요한 경우:")
        print("   - yolov8-lp-yasir (6.24MB): 가장 작은 크기, 빠른 추론")
        print("   - yolov5n-lp (3.8MB): 초경량 모델")
        
        print("\n2. 정확도가 중요한 경우:")
        print("   - yolov8m-lp-mkgoud (43MB): 균형잡힌 성능")
        print("   - yolov8m-lp-koushim (43MB): 높은 정확도")
        
        print("\n3. ANPR(자동 번호판 인식)이 필요한 경우:")
        print("   - yolov5m-anpr (40MB): 번호판 인식 특화")
        print("   - yolov5s-anpr (14MB): 경량 ANPR 모델")

        print("\n=== 주의사항 ===")
        print("1. 모델 크기가 클수록 더 높은 정확도를 제공하지만, 더 많은 GPU 메모리가 필요합니다.")
        print("2. 실시간 처리가 필요한 경우 작은 크기의 모델을 선택하세요.")
        print("3. 정확도가 중요한 경우 중간 크기 이상의 모델을 선택하세요.")
        print("4. ANPR이 필요한 경우 'anpr'가 포함된 모델을 선택하세요.")

    def parse_args(self):
        """명령행 인자를 파싱합니다."""
        parser = argparse.ArgumentParser(
            description='License Plate Detection using YOLO models',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
HuggingFace 토큰 설정:
1. 환경 변수로 설정 (권장):
   Linux/macOS: export HF_TOKEN="your_token_here"
   Windows: set HF_TOKEN=your_token_here

2. 명령행 인자로 설정:
   -t 또는 --token 옵션 사용

사용 예시:
1. 기본 사용:
   python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir

2. 다른 모델 사용:
   python license_plate_labeler.py -i input.jpg -o output -m yolov8m-lp-mkgoud

3. CPU 모드로 실행:
   python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir --device cpu

4. 시각화 없이 실행:
   python license_plate_labeler.py -i input.jpg -o output -m yolov8-lp-yasir --no-vis
"""
        )
        parser.add_argument('-i', '--input', required=True,
                          help='입력 이미지 또는 디렉토리 경로')
        parser.add_argument('-o', '--output', required=True,
                          help='결과를 저장할 디렉토리 경로')
        parser.add_argument('-m', '--model', default='yolov8-lp-yasir',
                          help='사용할 모델 선택 (기본값: yolov8-lp-yasir)')
        parser.add_argument('-t', '--token',
                          help='HuggingFace 토큰 (HF_TOKEN 환경 변수로도 설정 가능)')
        parser.add_argument('-c', '--confidence', type=float, default=0.5,
                          help='신뢰도 임계값 (0.0-1.0, 기본값: 0.5)')
        parser.add_argument('--max-size', type=int, default=800,
                          help='처리할 최대 이미지 크기 (기본값: 800)')
        parser.add_argument('-e', '--undetected-dir',
                          help='탐지되지 않은 이미지를 저장할 디렉토리 경로')
        parser.add_argument('--device', default='cuda',
                          help='사용할 디바이스 (cuda 또는 cpu, 기본값: cuda)')
        parser.add_argument('--no-vis', action='store_true',
                          help='시각화 결과를 저장하지 않음')
        parser.add_argument('--list-models', action='store_true',
                          help='사용 가능한 모델 목록 표시')
        return parser.parse_args()

    def _check_model_availability(self, model_name):
        """HuggingFace에서 모델 존재 여부 확인"""
        try:
            api_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(api_url, timeout=10)
            available = response.status_code == 200
            if available:
                print(f"✅ 모델 존재 확인: {model_name}")
            else:
                print(f"❌ 모델 존재하지 않음: {model_name}")
            return available
        except Exception as e:
            print(f"⚠️ 모델 존재 확인 실패: {e}")
            return False

    def _get_fallback_models(self, original_key):
        """원본 모델 실패시 시도할 대체 모델 목록 생성"""
        fallbacks = []
        
        # 1. 모델 정의에 fallback이 있는 경우
        if original_key in self.AVAILABLE_MODELS:
            original_info = self.AVAILABLE_MODELS[original_key]
            if 'fallback' in original_info:
                fallbacks.append(original_info['fallback'])
        
        # 2. 프레임워크별 추천 대체 모델
        if original_key in self.AVAILABLE_MODELS:
            framework = self.AVAILABLE_MODELS[original_key]['framework']
            if framework == "ultralytics":
                fallbacks.extend([
                    "yolov5m", "yolov8s"
                ])
            elif framework == "transformers":
                fallbacks.extend([
                    "yolos-small", "yolos-rego", "detr-resnet50",
                    "yolos-base", "detr-resnet-50"
                ])
        
        # 3. 검증된 모델들 추가
        verified_models = [k for k, v in self.AVAILABLE_MODELS.items() 
                          if v.get("verified", False) and k != original_key]
        fallbacks.extend(verified_models)
        
        # 중복 제거하고 항상 사용 가능한 모델을 마지막에 추가
        unique_fallbacks = []
        for fb in fallbacks:
            if fb not in unique_fallbacks and fb != original_key:
                unique_fallbacks.append(fb)
        
        # 항상 사용 가능한 모델을 마지막 보루로 추가
        always_available = [k for k, v in self.AVAILABLE_MODELS.items() 
                           if v.get("always_available", False)]
        for aa in always_available:
            if aa not in unique_fallbacks:
                unique_fallbacks.append(aa)
        
        return unique_fallbacks

    def _try_load_model(self, model_key):
        """단일 모델 로딩 시도"""
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"모델 키 '{model_key}'가 존재하지 않습니다.")
        
        model_info = self.AVAILABLE_MODELS[model_key]
        framework = model_info["framework"]
        
        print(f"🔄 모델 로딩 시도: {model_key}")
        print(f"   설명: {model_info['description']}")
        
        try:
            if framework == "ultralytics":
                return self._load_ultralytics_model(model_key), framework
            elif framework == "transformers":
                return self._load_transformers_model(model_info, model_key), framework
            else:
                raise ValueError(f"지원하지 않는 프레임워크: {framework}")
        except Exception as e:
            print(f"❌ 모델 '{model_key}' 로딩 실패: {e}")
            raise

    def _load_ultralytics_model(self, model_key):
        """Ultralytics 모델 로드"""
        try:
            model_info = self.AVAILABLE_MODELS[model_key]
            model_path = self._get_model_cache_path(model_key)
            
            if not os.path.exists(model_path):
                self._download_model_if_needed(model_info['download_uri'], model_key)
            
            self.logger.info(f"캐시된 모델 파일 사용: {model_path}")
            
            if model_info['processor_type'] in ['YOLOv8', 'YOLOv11']:
                from ultralytics import YOLO
                self.logger.info(f"{model_info['processor_type']} 모델 로드 시작")
                model = YOLO(model_path)
                # 모델 설정
                model.conf = 0.25  # 기본 신뢰도 임계값
                model.iou = 0.45   # IoU 임계값
                model.max_det = 100  # 최대 탐지 수
                # 클래스 이름 확인
                if hasattr(model, 'names'):
                    self.logger.info(f"모델 클래스: {model.names}")
                    self.logger.info(f"사용 가능한 클래스 수: {len(model.names)}")
                else:
                    self.logger.warning("모델에 names 속성이 없습니다")
                return model
            else:
                import torch
                self.logger.info("YOLOv5 모델 로드 시작")
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                model.conf = 0.25
                model.iou = 0.45
                model.max_det = 100
                return model
                
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _download_model_if_needed(self, url, model_key):
        """모델 파일 다운로드 (필요한 경우)"""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "license_plate_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 파일명 추출 및 모델 키 포함
        original_filename = os.path.basename(urlparse(url).path)
        if not original_filename:
            original_filename = f"{model_key}.pt"
        
        # 파일명에 모델 키 추가
        filename = f"{model_key}_{original_filename}"
        local_path = Path(os.path.join(cache_dir, filename))
        
        # 파일이 이미 존재하는지 확인
        if local_path.exists():
            self.logger.info(f"캐시된 모델 파일 사용: {local_path}")
            return str(local_path)
        
        # 다운로드 시도
        try:
            self.logger.info(f"모델 다운로드 시작: {url}")
            self.logger.info(f"저장 경로: {local_path}")
            
            # 다운로드 진행률 표시 함수
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percentage = 100. * block_num * block_size / total_size
                    self.logger.info(f"다운로드 진행률: {percentage:.1f}%")
            
            # 다운로드 시도
            urllib.request.urlretrieve(url, str(local_path), reporthook=show_progress)
            self.logger.info("모델 다운로드 완료")
            
            # 파일 크기 확인
            if not local_path.exists():
                raise ValueError(f"다운로드된 파일이 존재하지 않습니다: {local_path}")
            
            file_size = local_path.stat().st_size
            if file_size == 0:
                raise ValueError("다운로드된 파일이 비어있습니다")
            
            self.logger.info(f"다운로드된 파일 크기: {file_size / 1024 / 1024:.2f}MB")
            
            return str(local_path)
        except Exception as e:
            self.logger.error(f"모델 다운로드 중 에러 발생: {str(e)}")
            if local_path.exists():
                local_path.unlink()  # 실패한 경우 부분적으로 다운로드된 파일 삭제
            raise

    def _load_transformers_model(self, model_info, model_key):
        """Transformers 모델 로드"""
        try:
            model_name = model_info["name"]
            model = None
            processor = None
            
            # 캐시 디렉토리 설정
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 모델 캐시 경로
            model_cache_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
            
            # 캐시된 모델이 있는지 확인
            if os.path.exists(model_cache_path):
                self.logger.info(f"캐시된 모델 사용: {model_cache_path}")
                try:
                    model = AutoModelForObjectDetection.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True
                    )
                    processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True
                    )
                    self.logger.info("캐시된 모델 로드 성공")
                except Exception as e:
                    self.logger.warning(f"캐시된 모델 로드 실패: {str(e)}")
                    self.logger.info("원격 모델 다운로드 시도...")
            
            # 캐시된 모델이 없거나 로드 실패한 경우
            if model is None:
                # HuggingFace 토큰 설정
                if self.hf_token:
                    self.logger.info("HuggingFace 토큰을 사용하여 인증합니다.")
                    try:
                        model = AutoModelForObjectDetection.from_pretrained(
                            model_name,
                            token=self.hf_token,
                            cache_dir=cache_dir,
                            local_files_only=False,
                            resume_download=True
                        )
                        processor = AutoImageProcessor.from_pretrained(
                            model_name,
                            token=self.hf_token,
                            cache_dir=cache_dir,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        self.logger.error(f"토큰을 사용한 모델 로드 실패: {str(e)}")
                        self.logger.info("토큰 없이 재시도합니다...")
                        model = AutoModelForObjectDetection.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=False,
                            resume_download=True
                        )
                        processor = AutoImageProcessor.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            local_files_only=False,
                            resume_download=True
                        )
                else:
                    self.logger.warning("HuggingFace 토큰이 없습니다. 공개 모델만 다운로드 가능합니다.")
                    model = AutoModelForObjectDetection.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        resume_download=True
                    )
                    processor = AutoImageProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=False,
                        resume_download=True
                    )
            
            # 모델을 디바이스로 이동
            model.to(self.device)
            self.logger.info(f"Transformers 모델 로드 완료: {model_name}")
            
            return model, processor
            
        except Exception as e:
            self.logger.error(f"Transformers 모델 로드 중 오류 발생: {str(e)}")
            self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
            raise

    def _load_yolo_model(self, model_info, model_key):
        """
        YOLOv5/YOLOv8 모델 로드
        
        Args:
            model_info (dict): 모델 정보
            model_key (str): 모델 키
            
        Returns:
            model: 로드된 모델
        """
        processor_type = model_info["processor_type"]
        model_name = model_info["name"]
        
        # 로컬 경로 확인
        local_path = model_info.get("local_path")
        if local_path and os.path.exists(local_path):
            print(f"로컬 모델 파일 사용: {local_path}")
            model_path = local_path
        else:
            # 직접 다운로드가 필요한 경우
            if model_info.get("direct_download", False):
                download_uri = model_info["download_uri"]
                cache_path = self._get_model_cache_path(model_key)
                
                # 캐시된 모델이 없거나 손상된 경우 다운로드
                if not os.path.exists(cache_path):
                    print(f"캐시된 모델이 없습니다. 다운로드를 시작합니다.")
                    self._download_model_from_url(download_uri, cache_path)
                else:
                    print(f"캐시된 모델 사용: {cache_path}")
                
                model_path = cache_path
            else:
                # 기존 방식 (HuggingFace Hub 또는 모델명)
                model_path = model_name
        
        try:
            if processor_type == "YOLOv5":
                print("YOLOv5 모델 로드 중...")
                
                # 로컬 파일인 경우와 원격 모델인 경우 구분
                if os.path.exists(model_path):
                    # 로컬 파일 로드
                    print(f"로컬 YOLOv5 모델 로드: {model_path}")
                    import torch
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
                else:
                    # 원격 모델 로드 (기존 방식)
                    import torch
                    model = torch.hub.load('ultralytics/yolov5', model_path, force_reload=True, trust_repo=True)
                
                self.processor = None  # YOLOv5는 별도 프로세서 불필요
                return model
                
            elif processor_type == "YOLOv8":
                print("YOLOv8 모델 로드 중...")
                from ultralytics import YOLO

                # 로컬 파일 확인
                if os.path.exists(model_path):
                    print(f"로컬 YOLOv8 모델 로드: {model_path}")
                    model = YOLO(model_path)
                else:
                    print(f"로컬 모델을 찾을 수 없습니다: {model_path}")
                    print("기본 YOLOv8 모델을 사용합니다.")
                    model = YOLO(model_name)

                self.processor = None  # YOLOv8도 별도 프로세서 불필요
                return model

            elif processor_type == "YOLOv11":
                print("YOLOv11 모델 로드 중...")
                from ultralytics import YOLO
                
                # 로컬 파일 확인
                if os.path.exists(model_path):
                    print(f"로컬 YOLOv11 모델 로드: {model_path}")
                    model = YOLO(model_path)
                else:
                    print(f"로컬 모델을 찾을 수 없습니다: {model_path}")
                    print("기본 YOLOv11 모델을 사용합니다.")
                    model = YOLO(model_name)

                self.processor = None  # YOLOv11도 별도 프로세서 불필요
                return model

            else:
                raise ValueError(f"지원하지 않는 YOLO 타입: {processor_type}")
                
        except ImportError as e:
            print(f"YOLO 라이브러리 로드 실패: {e}")
            print("ultralytics 설치 필요: pip install ultralytics")
            print("대신 transformers 기반 모델을 사용하세요: -m yolos-small")
            raise
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            
            # 캐시 파일이 손상된 경우 재다운로드 시도
            if model_info.get("direct_download", False) and os.path.exists(model_path):
                print("캐시된 모델이 손상된 것 같습니다. 재다운로드를 시도합니다.")
                try:
                    os.remove(model_path)
                    self._download_model_from_url(model_info["download_uri"], model_path)
                    
                    # 재시도
                    if processor_type == "YOLOv5":
                        import torch
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
                    else:  # YOLOv8
                        from ultralytics import YOLO
                        model = YOLO(model_path)
                    
                    return model
                    
                except Exception as retry_error:
                    print(f"재다운로드도 실패: {retry_error}")
            
            raise

    def _load_with_fallbacks(self, original_key):
        """대체 모델 시스템을 사용한 모델 로딩"""
        # 원본 모델 시도
        try:
            model, framework = self._try_load_model(original_key)
            return model, framework
        except Exception as original_error:
            print(f"🔄 원본 모델 '{original_key}' 실패, 대체 모델을 시도합니다...")
            print(f"   원본 오류: {original_error}")
        
        # 대체 모델들 시도
        fallback_models = self._get_fallback_models(original_key)
        
        for fallback_key in fallback_models:
            try:
                print(f"🔄 대체 모델 시도: {fallback_key}")
                model, framework = self._try_load_model(fallback_key)
                print(f"✅ 대체 모델 성공: {fallback_key}")
                self.model_key = fallback_key  # 성공한 모델로 키 업데이트
                return model, framework
            except Exception as fallback_error:
                print(f"❌ 대체 모델 '{fallback_key}' 실패: {fallback_error}")
                continue
        
        # 모든 대체 모델 실패시 최후의 수단
        raise Exception(
            f"원본 모델 '{original_key}' 및 모든 대체 모델 로딩에 실패했습니다. "
            "다음을 확인해주세요:\n"
            "1. 인터넷 연결 상태\n"
            "2. 필수 라이브러리 설치: pip install ultralytics transformers\n"
            "3. --list-models로 사용 가능한 모델 확인"
        )

    def _get_model_cache_path(self, model_key):
        """
        모델 캐시 경로 생성
        
        Args:
            model_key (str): 모델 키
            
        Returns:
            str: 캐시 파일 경로
        """
        cache_dir = os.path.expanduser("~/.cache/license_plate_models")
        model_info = self.AVAILABLE_MODELS[model_key]
        
        # 원본 파일명 추출
        if 'model_file' in model_info:
            original_filename = model_info['model_file']
        else:
            original_filename = f"{model_key}.pt"
        
        # 모델 키를 포함한 파일명 생성
        filename = f"{model_key}_{original_filename}"
        return os.path.join(cache_dir, filename)

    def _download_model_from_url(self, url: str, local_path: Path) -> None:
        """URL에서 모델 파일 다운로드"""
        try:
            # local_path가 문자열인 경우 Path 객체로 변환
            if isinstance(local_path, str):
                local_path = Path(local_path)
            
            # 디렉토리 생성
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Hugging Face URL인 경우 토큰 추가
            if "huggingface.co" in url:
                if not self.hf_token:
                    raise ValueError("Hugging Face 토큰이 설정되지 않았습니다. HF_TOKEN 환경 변수를 설정하거나 토큰을 직접 입력해주세요.")
                
                # URL에 토큰 추가
                if "?" not in url:
                    url = f"{url}?token={self.hf_token}"
                else:
                    url = f"{url}&token={self.hf_token}"
                
                self.logger.info(f"Hugging Face URL에 토큰이 추가되었습니다: {url}")
            
            # 다운로드 시작
            self.logger.info(f"모델 다운로드 시작: {url}")
            self.logger.info(f"저장 경로: {local_path}")
            
            # 다운로드 진행률 표시 함수
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percentage = 100. * block_num * block_size / total_size
                    self.logger.info(f"다운로드 진행률: {percentage:.1f}%")
            
            # 다운로드 시도
            try:
                urllib.request.urlretrieve(url, str(local_path), reporthook=show_progress)
                self.logger.info("모델 다운로드 완료")
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise ValueError("Hugging Face 토큰이 유효하지 않습니다. 토큰을 확인해주세요.") from e
                elif e.code == 404:
                    raise ValueError(f"모델 파일을 찾을 수 없습니다: {url}") from e
                else:
                    raise ValueError(f"모델 다운로드 실패 (HTTP {e.code}): {e.reason}") from e
            except urllib.error.URLError as e:
                raise ValueError(f"모델 다운로드 실패 (URL 에러): {str(e)}") from e
            
            # 파일 크기 확인
            if not local_path.exists():
                raise ValueError(f"다운로드된 파일이 존재하지 않습니다: {local_path}")
            
            file_size = local_path.stat().st_size
            if file_size == 0:
                raise ValueError("다운로드된 파일이 비어있습니다")
            
            self.logger.info(f"다운로드된 파일 크기: {file_size / 1024 / 1024:.2f}MB")
            
        except Exception as e:
            self.logger.error(f"모델 다운로드 중 에러 발생: {str(e)}")
            if local_path.exists():
                local_path.unlink()  # 실패한 경우 부분적으로 다운로드된 파일 삭제
            raise

    def __init__(self, model_key, token=None, max_size=640):
        """초기화"""
        # 로거 초기화를 가장 먼저 수행
        self.logger = logging.getLogger(__name__)
        
        self.model_key = model_key
        self.token = token
        self.max_size = max_size
        self.hf_token = token or os.getenv('HF_TOKEN')
        
        # CUDA 사용 가능 여부 확인 후 디바이스 설정
        if torch.cuda.is_available():
            self.device = 0  # 첫 번째 CUDA 디바이스 사용
            torch.cuda.empty_cache()  # CUDA 메모리 정리
            self.logger.info(f"CUDA 디바이스 사용: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            self.logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        
        # 모델 정보 로깅
        self.logger.info(f"사용 디바이스: {self.device}")
        self.logger.info(f"선택된 모델: {model_key}")
        
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"지원하지 않는 모델입니다: {model_key}")
        
        model_info = self.AVAILABLE_MODELS[model_key]
        self.logger.info(f"모델 설명: {model_info['description']}")
        self.logger.info(f"프레임워크: {model_info['framework']}")
        self.logger.info(f"프로세서 타입: {model_info['processor_type']}")
        
        # 모델 프레임워크와 프로세서 타입 설정
        self.model_framework = model_info['framework']
        self.processor_type = model_info['processor_type']
        
        # 모델 로드
        try:
            if self.model_framework == "ultralytics":
                self.model = self._load_ultralytics_model(model_key)
            else:  # transformers
                self.model, self.processor = self._load_transformers_model(model_info, model_key)
        except Exception as e:
            self.logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
            self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
            raise

    def _save_cropped_plate(self, image, bbox, output_path, confidence=None):
        """번호판 영역만 추출하여 저장"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # 이미지 경계 확인
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # 번호판 영역 추출
            plate_region = image[y1:y2, x1:x2].copy()  # 복사본 생성
            
            if plate_region.size > 0:  # 유효한 영역인 경우에만 저장
                cv2.imwrite(output_path, plate_region)
                self.logger.info(f"번호판 영역 저장 (신뢰도: {confidence:.2f}): {output_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"번호판 영역 저장 중 오류 발생: {str(e)}")
            return False

    def _draw_detection_info(self, image, bbox, confidence):
        """탐지 정보를 이미지에 표시"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            # 이미지 경계 확인
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # 붉은색 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 신뢰도 텍스트
            conf_text = f"Conf: {confidence:.2f}"
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width + 4, y1), (0, 0, 0), -1)
            # 텍스트
            cv2.putText(image, conf_text, (x1 + 2, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            return True
        except Exception as e:
            self.logger.error(f"탐지 정보 표시 중 오류 발생: {str(e)}")
            return False

    def process_single_image(self, image_path, output_dir, confidence_threshold=0.5, save_visualization=True, undetected_dir=None):
        """단일 이미지 처리"""
        try:
            # 이미지 로드
            self.logger.info(f"이미지 로드 중: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # 이미지 크기 로깅
            height, width = image.shape[:2]
            self.logger.info(f"이미지 크기: {width}x{height}")
            
            # 모델 추론
            self.logger.info("모델 추론 시작")
            try:
                if isinstance(self.model, torch.nn.Module):  # YOLOv5
                    results = self.model(image)
                    detections = self._process_ultralytics_results(results, width, height, confidence_threshold)
                elif self.processor_type == 'YOLOv8':  # YOLOv8
                    detections = self._process_yolov8_inference(image, confidence_threshold)
                elif self.processor_type == 'YOLOv11':  # YOLOv11
                    detections = self._process_yolov11_inference(image, confidence_threshold)
                else:
                    raise ValueError(f"지원하지 않는 프로세서 타입입니다: {self.processor_type}")
            except Exception as e:
                self.logger.error(f"모델 추론 실패: {str(e)}")
                self.logger.info("CPU로 재시도합니다...")
                try:
                    if isinstance(self.model, torch.nn.Module):  # YOLOv5
                        results = self.model(image)
                        detections = self._process_ultralytics_results(results, width, height, confidence_threshold)
                    elif self.processor_type == 'YOLOv8':  # YOLOv8
                        detections = self._process_yolov8_inference(image, confidence_threshold)
                    elif self.processor_type == 'YOLOv11':  # YOLOv11
                        detections = self._process_yolov11_inference(image, confidence_threshold)
                    else:
                        raise ValueError(f"지원하지 않는 프로세서 타입입니다: {self.processor_type}")
                except Exception as e:
                    self.logger.error(f"CPU 추론도 실패: {str(e)}")
                    raise

            # 탐지 결과 처리
            if not detections:
                self.logger.warning(f"번호판을 찾을 수 없습니다: {os.path.basename(image_path)}")
                if undetected_dir:
                    undetected_path = os.path.join(undetected_dir, os.path.basename(image_path))
                    cv2.imwrite(undetected_path, image)
                return False
            
            # 원본 이미지에 탐지 정보 표시
            image_with_detections = image.copy()
            for detection in detections:
                self._draw_detection_info(image_with_detections, detection['bbox'], detection['confidence'])
            
            # 결과 저장
            if save_visualization:
                self._save_visualization(image_with_detections, detections, os.path.join(output_dir, os.path.basename(image_path)))
            
            # YOLO 형식 라벨 저장
            label_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
            self._save_yolo_labels(detections, label_path)
            
            # YOLOv5 모델인 경우 번호판 영역만 저장
            if isinstance(self.model, torch.nn.Module):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                for i, detection in enumerate(detections):
                    plate_path = os.path.join(output_dir, f"{base_name}_plate_{i+1}.jpg")
                    self._save_cropped_plate(image, detection['bbox'], plate_path, detection['confidence'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _process_ultralytics_results(self, results, width, height, confidence_threshold):
        """Ultralytics 모델 결과 처리"""
        detections = []
        
        # YOLOv5 모델 결과 처리
        if hasattr(results, 'xyxy'):
            boxes = results.xyxy[0].cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if conf >= confidence_threshold:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls)
                    })
        # YOLOv8/YOLOv11 모델 결과 처리
        elif hasattr(results, 'boxes'):
            boxes = results.boxes
            for box in boxes:
                if box.conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf),
                        'class': int(box.cls)
                    })
        
        return detections

    def _process_transformers_results(self, image, confidence_threshold):
        """Transformers 모델 결과 처리"""
        detections = []
        
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        # 이미지 크기
        height, width = image.shape[:2]
        
        # 탐지 결과 변환
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # 신뢰도가 임계값보다 높은 경우만 처리
            if score >= confidence_threshold:
                # 바운딩 박스 좌표 추출
                x1, y1, x2, y2 = box.tolist()
                
                # 좌표가 이미지 범위 내에 있는지 확인
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                # 최소 크기 확인 (20x20 픽셀)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    self.logger.warning(f"탐지된 영역이 너무 작습니다: {x2-x1:.1f}x{y2-y1:.1f} 픽셀")
                    continue
                
                # 정규화된 좌표로 변환
                x1_norm = x1 / width
                y1_norm = y1 / height
                x2_norm = x2 / width
                y2_norm = y2 / height
                
                detections.append({
                    'bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                    'confidence': float(score),
                    'class': 0  # 번호판 클래스
                })
        
        return detections

    def visualize_detections(self, image_path, detections, output_path=None):
        """
        탐지 결과 시각화
        
        Args:
            image_path (str): 원본 이미지 경로
            detections (list): 탐지 결과
            output_path (str): 시각화 결과 저장 경로 (선택사항)
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x_min, y_min, x_max, y_max = map(int, bbox)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 신뢰도 텍스트 추가
            label = f"License Plate: {confidence:.3f}"
            cv2.putText(image_rgb, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"시각화 결과 저장: {output_path}")
        
        return image_rgb
    
    def process_directory(self, input_dir, output_dir, confidence_threshold=0.3, save_visualization=True, undetected_dir=None):
        """디렉토리 내 모든 이미지 처리"""
        try:
            # 입력 디렉토리 확인
            if not os.path.exists(input_dir):
                raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 미탐지 디렉토리 생성
            if undetected_dir:
                os.makedirs(undetected_dir, exist_ok=True)
            
            # 이미지 파일 목록
            try:
                image_files = [f for f in os.listdir(input_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            except Exception as e:
                raise ValueError(f"입력 디렉토리 읽기 실패: {input_dir} - {str(e)}")
            
            if not image_files:
                self.logger.warning(f"처리할 이미지가 없습니다: {input_dir}")
                return
            
            self.logger.info(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
            
            # 각 이미지 처리
            processed_count = 0
            failed_count = 0
            
            for i, image_file in enumerate(image_files, 1):
                self.logger.info(f"[{i}/{len(image_files)}] {image_file} 처리 중...")
                image_path = os.path.join(input_dir, image_file)
                
                try:
                    detections = self.process_single_image(
                        image_path=image_path,
                        output_dir=output_dir,
                        confidence_threshold=confidence_threshold,
                        save_visualization=save_visualization,
                        undetected_dir=undetected_dir
                    )
                    
                    if detections:
                        processed_count += 1
                    else:
                        failed_count += 1
                        self.logger.warning(f"번호판을 찾을 수 없습니다: {image_file}")
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    continue
            
            self.logger.info(f"처리 완료: 총 {len(image_files)}개 중 {processed_count}개 성공, {failed_count}개 실패")
            
        except Exception as e:
            self.logger.error(f"디렉토리 처리 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _save_visualization(self, image, detections, output_path):
        """탐지 결과 시각화 저장"""
        try:
            # 이미지 복사
            vis_image = image.copy()
            
            # 이미지 크기
            height, width = vis_image.shape[:2]
            
            # 각 탐지 결과에 대해 박스 그리기
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                
                # 픽셀 좌표로 변환
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # 박스 그리기
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 신뢰도 텍스트
                text = f"{conf:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 결과 저장
            cv2.imwrite(output_path, vis_image)
            
        except Exception as e:
            self.logger.error(f"시각화 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _save_yolo_labels(self, detections, output_path):
        """YOLO 형식 라벨 파일 저장"""
        try:
            with open(output_path, 'w') as f:
                for det in detections:
                    bbox = det['bbox']
                    # YOLO 형식으로 변환 (x_center, y_center, width, height)
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    
                    # 클래스 ID는 0 (번호판)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            
            self.logger.info(f"YOLO 라벨 저장: {output_path}")
            
        except Exception as e:
            self.logger.error(f"YOLO 라벨 저장 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())

def parse_args():
    parser = argparse.ArgumentParser(description='License Plate Detection and Labeling Tool')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing images')
    parser.add_argument('-o', '--output', required=True, help='Output directory for labeled images')
    parser.add_argument('-t', '--token', help='HuggingFace token for model download')
    parser.add_argument('-c', '--confidence', type=float, default=0.6, help='Confidence threshold (default: 0.6)')
    parser.add_argument('--max-size', type=int, default=640, help='Maximum image size for processing (default: 640)')
    parser.add_argument('-u', '--undetected', help='Directory to save undetected images')
    parser.add_argument('-m', '--model', default='yolov11n', help='Model to use (default: yolov11n)')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    return parser.parse_args()

def main():
    try:
        # 명령줄 인자 파싱
        args = parse_args()
        
        # 모델 목록 출력
        if args.list_models:
            print("\n사용 가능한 모델 목록:")
            print("-" * 80)
            for key, info in LicensePlateYOLOLabeler.AVAILABLE_MODELS.items():
                print(f"모델 키: {key}")
                print(f"이름: {info['name']}")
                print(f"설명: {info['description']}")
                print(f"프레임워크: {info['framework']}")
                print(f"크기: {info['size']}")
                print(f"성능: {info['performance']}")
                print(f"검증됨: {info['verified']}")
                print("-" * 80)
            return
        
        # 입력/출력 디렉토리 확인
        if not os.path.exists(args.input):
            print(f"오류: 입력 디렉토리가 존재하지 않습니다: {args.input}")
            return
        
        # 모델 초기화
        try:
            labeler = LicensePlateYOLOLabeler(
                model_key=args.model,
                token=args.token,
                max_size=args.max_size
            )
        except Exception as e:
            print(f"오류: 모델 초기화 중 예외 발생: {str(e)}")
            return
        
        # 디렉토리 처리
        try:
            labeler.process_directory(
                input_dir=args.input,
                output_dir=args.output,
                confidence_threshold=args.confidence,
                save_visualization=True,
                undetected_dir=args.undetected
            )
        except Exception as e:
            print(f"오류: 디렉토리 처리 중 예외 발생: {str(e)}")
            return
            
    except Exception as e:
        print(f"오류: {str(e)}")
        return

if __name__ == "__main__":
    main()