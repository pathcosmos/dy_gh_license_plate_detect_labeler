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
    DetrForObjectDetection
)
from ultralytics import YOLO
from typing import Any, List, Dict, Tuple

class LicensePlateYOLOLabeler:
    # 사용 가능한 모델들 정의 (기존 모델 + 검증된 추가 모델)
    AVAILABLE_MODELS = {
        # YOLOS 기반 모델 (Vision Transformer) - 번호판 전용 파인튜닝
        "yolos-small-finetuned": {
            "name": "nickmuchi/yolos-small-finetuned-license-plate-detection",
            "description": "YOLO + Vision Transformer, 번호판 전용 파인튜닝 (90MB)",
            "download_uri": "https://huggingface.co/nickmuchi/yolos-small-finetuned-license-plate-detection",
            "processor_type": "YolosImageProcessor",
            "framework": "transformers",
            "size": "90MB",
            "performance": "높음",
            "pros": ["높은 정확도", "Transformer 기반", "빠른 추론"],
            "cons": ["상대적으로 큰 모델 크기"],
            "verified": True
        },
        "yolos-rego": {
            "name": "nickmuchi/yolos-small-rego-plates-detection",
            "description": "차량+번호판 동시 탐지, 735 이미지로 200 에포크 훈련 (90MB)",
            "download_uri": "https://huggingface.co/nickmuchi/yolos-small-rego-plates-detection",
            "processor_type": "YolosImageProcessor",
            "framework": "transformers",
            "size": "90MB",
            "performance": "높음",
            "classes": ["vehicle", "license-plate"],
            "pros": ["차량과 번호판 동시 탐지", "좋은 일반화 성능"],
            "cons": ["데이터셋이 상대적으로 작음"],
            "verified": True
        },
        
        "yolov11x": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11x 모델, 최고 정확도 (114MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1x.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "114MB",
            "performance": "최고",
            "metrics": {"precision": 0.9893, "recall": 0.9508, "mAP@50": 0.9813, "mAP@50-95": 0.7260},
            "pros": ["최고 정확도", "강력한 특징 추출", "복잡한 케이스 처리 우수"],
            "cons": ["매우 큰 모델 크기", "높은 GPU 메모리 요구사항"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1x.pt",
        },
        "yolov11l": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 large 모델, 매우 높은 정확도 (51.2MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1l.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "51.2MB",
            "performance": "매우 높음",
            "metrics": {"precision": 0.985, "recall": 0.948, "mAP@50": 0.978, "mAP@50-95": 0.72},
            "pros": ["매우 높은 정확도", "강력한 특징 추출", "실시간 처리 가능"],
            "cons": ["큰 모델 크기"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1l.pt",
        },
        "yolov11m": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 medium 모델, 높은 정확도 (40.5MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1m.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "40.5MB",
            "performance": "높음",
            "metrics": {"precision": 0.98, "recall": 0.95, "mAP@50": 0.97, "mAP@50-95": 0.70},
            "pros": ["높은 정확도", "적절한 추론 속도", "실시간 처리 가능"],
            "cons": ["x 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1m.pt",
        },
        "yolov11s": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 small 모델, 균형잡힌 성능 (19.2MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "19.2MB",
            "performance": "중간",
            "metrics": {"precision": 0.97, "recall": 0.94, "mAP@50": 0.96, "mAP@50-95": 0.68},
            "pros": ["균형잡힌 성능", "적절한 모델 크기", "실시간 처리 가능"],
            "cons": ["x 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1s.pt",
        },
        "yolov11n": {
            "name": "morsetechlab/yolov11-license-plate-detection",
            "description": "YOLOv11 nano 모델, 가장 작은 크기 (5.47MB)",
            "download_uri": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt",
            "processor_type": "YOLOv11",
            "framework": "ultralytics",
            "size": "5.47MB",
            "performance": "중간",
            "metrics": {"precision": 0.95, "recall": 0.92, "mAP@50": 0.94, "mAP@50-95": 0.65},
            "pros": ["매우 작은 모델 크기", "빠른 추론", "저사양 기기 적합"],
            "cons": ["다른 버전 대비 정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "AGPLv3",
            "model_file": "license-plate-finetune-v1n.pt",
        },
        # YOLOS 모델
        "yolos-small": {
            "name": "hustvl/yolos-small",
            "description": "YOLOS small 모델, 경량화된 버전 (14MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-small/resolve/main/pytorch_model.bin",
            "processor_type": "YolosImageProcessor",
            "framework": "transformers",
            "size": "14MB",
            "performance": "낮음",
            "verified": True,
        },
        "yolos-base": {
            "name": "hustvl/yolos-base",
            "description": "YOLOS base 모델, 균형잡힌 성능 (24MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-base/resolve/main/pytorch_model.bin",
            "processor_type": "YolosImageProcessor",
            "framework": "transformers",
            "size": "24MB",
            "performance": "중간",
            "verified": True,
        },
        "yolos-tiny": {
            "name": "hustvl/yolos-tiny",
            "description": "YOLOS tiny 모델, 초경량 버전 (6MB)",
            "download_uri": "https://huggingface.co/hustvl/yolos-tiny/resolve/main/pytorch_model.bin",
            "processor_type": "YolosImageProcessor",
            "framework": "transformers",
            "size": "6MB",
            "performance": "낮음",
            "verified": True,
        },
        # DETR 모델
        "detr-resnet-50": {
            "name": "facebook/detr-resnet-50",
            "description": "DETR ResNet-50 모델, 기본 버전 (159MB)",
            "download_uri": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/pytorch_model.bin",
            "processor_type": "DetrImageProcessor",
            "framework": "transformers",
            "size": "159MB",
            "performance": "높음",
            "verified": True,
        },
        "detr-resnet-101": {
            "name": "facebook/detr-resnet-101",
            "description": "DETR ResNet-101 모델, 고성능 버전 (232MB)",
            "download_uri": "https://huggingface.co/facebook/detr-resnet-101/resolve/main/pytorch_model.bin",
            "processor_type": "DetrImageProcessor",
            "framework": "transformers",
            "size": "232MB",
            "performance": "매우 높음",
            "verified": True,
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
            "metrics": {"precision": 0.92, "recall": 0.89, "mAP@50": 0.91},
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "Apache-2.0",
            "model_file": "best.pt",
        },
        "yolov8-lp-koushim": {
            "name": "Koushim/yolov8-license-plate-detection",
            "description": "YOLOv8 nano 번호판 탐지 모델 (Koushim, 6.25MB)",
            "download_uri": "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
            "processor_type": "YOLOv8",
            "framework": "ultralytics",
            "size": "6.25MB",
            "performance": "빠름",
            "metrics": {"precision": 0.93, "recall": 0.90, "mAP@50": 0.92},
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt",
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
            "metrics": {"precision": 0.94, "recall": 0.91, "mAP@50": 0.93},
            "pros": ["작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 중간"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "LP-detection.pt",
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
            "metrics": {"mAP@0.5": 0.978},
            "pros": ["매우 작은 크기", "빠른 추론", "실시간 처리 적합"],
            "cons": ["정확도 낮음"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt",
        },
        "yolov5s": {
            "name": "keremberke/yolov5s-license-plate",
            "description": "YOLOv5 small 모델, 경량화된 버전 (14MB)",
            "download_uri": "https://huggingface.co/keremberke/yolov5s-license-plate/resolve/main/best.pt",
            "processor_type": "YOLOv5",
            "framework": "ultralytics",
            "size": "14MB",
            "performance": "중간",
            "metrics": {"mAP@0.5": 0.985},
            "pros": ["적절한 크기", "균형잡힌 성능", "실시간 처리 가능"],
            "cons": ["nano 버전보다 느림"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt",
        },
        "yolov5m": {
            "name": "keremberke/yolov5m-license-plate",
            "description": "YOLOv5 medium 모델, 번호판 탐지 특화 (40MB)",
            "download_uri": "https://huggingface.co/keremberke/yolov5m-license-plate/resolve/main/best.pt",
            "processor_type": "YOLOv5",
            "framework": "ultralytics",
            "size": "40MB",
            "performance": "높음",
            "metrics": {"mAP@0.5": 0.988},
            "pros": ["높은 정확도", "안정적인 탐지", "복잡한 케이스 처리 우수"],
            "cons": ["큰 모델 크기", "추론 속도 느림"],
            "direct_download": True,
            "verified": True,
            "license": "MIT",
            "model_file": "best.pt",
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

    @staticmethod
    def list_available_models():
        """사용 가능한 모델 목록을 출력합니다."""
        print("\n=== 사용 가능한 모델 목록 ===\n")
        
        # 모델 카테고리별로 그룹화
        categories = {
            "🔥 최고 정확도 모델": ["yolov11x"],
            "💪 고성능 모델": ["yolov11l", "yolov11m", "detr-resnet50"],
            "⚖️ 균형잡힌 모델": ["yolov11s", "yolos-small", "yolos-rego"],
            "🚀 경량 모델": ["yolov11n", "yolov5m", "yolov8s"]
        }
        
        for category, model_keys in categories.items():
            print(f"\n{category}")
            print("=" * 50)
            
            for model_key in model_keys:
                if model_key in LicensePlateYOLOLabeler.AVAILABLE_MODELS:
                    model = LicensePlateYOLOLabeler.AVAILABLE_MODELS[model_key]
                    print(f"\n모델 키: {model_key}")
                    print(f"이름: {model['name']}")
                    print(f"설명: {model['description']}")
                    print(f"프레임워크: {model['framework']}")
                    print(f"크기: {model['size']}")
                    print(f"성능: {model['performance']}")
                    
                    if 'metrics' in model:
                        metrics = model.get('metrics', {})
                        if metrics:
                            print("\n성능 지표:")
                            for name, val in metrics.items():
                                if isinstance(val, float):
                                    print(f"  - {name}: {val:.4f}")
                                else:
                                    print(f"  - {name}: {val}")
                    
                    pros_list = model.get('pros', [])
                    print("\n장점:" if pros_list else "\n장점: 없음")
                    for pro in pros_list:
                        print(f"  - {pro}")
                    
                    cons_list = model.get('cons', [])
                    print("\n단점:" if cons_list else "\n단점: 없음")
                    for con in cons_list:
                        print(f"  - {con}")
                    
                    if 'license' in model:
                        print(f"\n라이선스: {model['license']}")
                    
                    print("\n사용 예시:")
                    print(f"  python license_plate_labeler.py --model {model_key} --input 이미지.jpg --output 결과")
                    print("-" * 50)
        
        print("\n=== 모델 선택 가이드 ===")
        print("1. 최고 정확도가 필요한 경우:")
        print("   - yolov11x: mAP@50 0.9813, 가장 높은 정확도")
        print("   - yolov11l: mAP@50 0.978, 높은 정확도와 적절한 크기")
        print("\n2. 균형잡힌 성능이 필요한 경우:")
        print("   - yolov11m: mAP@50 0.975, 중간 크기와 좋은 성능")
        print("   - yolov11s: mAP@50 0.972, 작은 크기와 양호한 성능")
        print("\n3. 경량화가 필요한 경우:")
        print("   - yolov11n: 5.47MB, 가장 작은 크기")
        print("   - yolov8s: 22MB, 빠른 처리 속도")
        print("\n4. 특수 사용 사례:")
        print("   - yolos-rego: 차량과 번호판 동시 탐지")
        print("   - yolos-small: Vision Transformer 기반 번호판 탐지")
        print("   - detr-resnet50: DETR 기반 고정밀 탐지")
        
        print("\n=== 주의사항 ===")
        print("1. YOLOv11 모델들은 ultralytics 패키지가 필요합니다.")
        print("2. YOLOS/DETR 모델들은 transformers 패키지가 필요합니다.")
        print("3. 모델 크기가 클수록 더 높은 정확도를 제공하지만, 더 많은 GPU 메모리가 필요합니다.")
        print("4. yolov11x 모델은 AGPLv3 라이선스로 제공됩니다.")

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
                return self._load_yolo_model(model_info, model_key), framework
            elif framework == "transformers":
                return self._load_transformers_model(model_info, model_key), framework
            else:
                raise ValueError(f"지원하지 않는 프레임워크: {framework}")
        except Exception as e:
            print(f"❌ 모델 '{model_key}' 로딩 실패: {e}")
            raise

    def _load_transformers_model(self, model_info, model_key):
        """Transformers 모델 로드"""
        model_name = model_info["name"]
        processor_type = model_info["processor_type"]
        
        # 모델 존재 여부 확인 (HuggingFace 모델인 경우)
        if "/" in model_name and not self._check_model_availability(model_name):
            raise Exception(f"Model {model_name} does not exist on HuggingFace")
        
        if processor_type == "DetrImageProcessor":
            if hasattr(self, 'token') and self.token:
                self.processor = DetrImageProcessor.from_pretrained(model_name, token=self.token)
                model = DetrForObjectDetection.from_pretrained(model_name, token=self.token)
            else:
                self.processor = DetrImageProcessor.from_pretrained(model_name)
                model = DetrForObjectDetection.from_pretrained(model_name)
        
        elif processor_type in ("YolosImageProcessor", "YOLOS"):
            if hasattr(self, 'token') and self.token:
                self.processor = YolosImageProcessor.from_pretrained(model_name, token=self.token)
                model = YolosForObjectDetection.from_pretrained(model_name, token=self.token)
            else:
                self.processor = YolosImageProcessor.from_pretrained(model_name)
                model = YolosForObjectDetection.from_pretrained(model_name)
        
        else:
            raise ValueError(f"지원하지 않는 프로세서 타입: {processor_type}")
        
        return model

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
            # if processor_type == "YOLOv5":
            #     print("ultralytics 패키지로 YOLOv5 모델 로드 중…")
            #     from ultralytics import YOLO
            #     # ultralytics.YOLO은 .pt 파일 경로를 직접 받습니다
            #     model = YOLO(model_path)
            #     self.processor = None
            #     return model
                
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
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"모델 키 '{model_key}'가 존재하지 않습니다.")
            
        model_info = self.AVAILABLE_MODELS[model_key]
        
        # 모델의 고유 식별자를 생성하기 위한 정보 수집
        model_name = model_info["name"]
        framework = model_info["framework"]
        processor_type = model_info["processor_type"]
        
        # 다운로드 URI에서 파일명 추출 (있는 경우)
        download_uri = model_info.get("download_uri", "")
        original_filename = ""
        if download_uri:
            parsed_url = urlparse(download_uri)
            original_filename = os.path.basename(parsed_url.path)
        
        # 모델 파일명이 명시적으로 지정된 경우 사용
        if "model_file" in model_info:
            original_filename = model_info["model_file"]
        
        # 고유한 캐시 파일명 생성
        if original_filename:
            # 원본 파일명이 있는 경우, 프레임워크와 프로세서 타입을 접두어로 추가
            cache_filename = f"{framework}_{processor_type}_{original_filename}"
        else:
            # 원본 파일명이 없는 경우, 모델 이름을 기반으로 생성
            model_name_safe = model_name.replace("/", "_").replace("\\", "_")
            cache_filename = f"{framework}_{processor_type}_{model_name_safe}.pt"
        
        # 캐시 디렉토리 생성
        cache_dir = os.path.expanduser("~/.cache/license_plate_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 최종 캐시 경로 반환
        cache_path = os.path.join(cache_dir, cache_filename)
        
        print(f"캐시 파일 경로: {cache_path}")
        print(f"  - 프레임워크: {framework}")
        print(f"  - 프로세서: {processor_type}")
        print(f"  - 원본 파일명: {original_filename if original_filename else 'N/A'}")
        
        return cache_path

    def _download_model_from_url(self, url, local_path):
        """
        URL에서 모델 파일을 다운로드
        
        Args:
            url (str): 다운로드할 모델 URL
            local_path (str): 저장할 로컬 경로
        """
        try:
            print(f"모델 다운로드 중: {url}")
            
            # SSL 컨텍스트 설정 (인증서 검증 우회)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 다운로드 진행률 표시 함수
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) // total_size)
                    print(f"\r다운로드 진행률: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='')
                else:
                    print(f"\r다운로드 중: {downloaded // (1024*1024)}MB", end='')
            
            # 다운로드 실행
            urllib.request.urlretrieve(url, local_path, reporthook=show_progress)
            print(f"\n모델 다운로드 완료: {local_path}")
            
        except Exception as e:
            print(f"\n모델 다운로드 실패: {e}")
            raise

    def __init__(self, model_key="yolos-small", local_model_path=None, force_cpu=False, token=None):
        """
        번호판 탐지 및 YOLO 라벨링 파일 생성기 초기화
        
        Args:
            model_key (str): 사용할 모델 키 (기본값: yolos-small)
            local_model_path (str): 로컬 모델 경로 (오프라인 사용시)
            force_cpu (bool): GPU 사용을 강제로 비활성화
            token (str): HuggingFace 액세스 토큰 (private 모델 접근시)
        """
        print("\n모델 초기화 중...")
        
        # 토큰 설정 (환경 변수 또는 인자에서 가져오기)
        self.token = token or os.getenv('HF_TOKEN')
        if self.token:
            print("HuggingFace 토큰이 설정되었습니다.")
        else:
            print("HuggingFace 토큰이 설정되지 않았습니다. 일부 모델에 접근이 제한될 수 있습니다.")
            print("토큰 설정 방법:")
            print("1. 환경 변수로 설정: export HF_TOKEN='your_token_here'")
            print("2. 명령줄 인자로 설정: --token 'your_token_here'")
        
        # 모델 정보 확인 및 기본값 설정
        if model_key not in self.AVAILABLE_MODELS and not local_model_path:
            print(f"지원하지 않는 모델 키: {model_key}")
            print("기본 안정 모델로 변경합니다: yolos-small")
            model_key = "yolos-small"
        
        # GPU 사용 가능 여부 확인 및 설정
        self.device = self._setup_device(force_cpu)
        print(f"사용 중인 디바이스: {self.device}")
        
        # 모델 프레임워크 및 키 추적 변수 초기화
        self.model_framework = None
        self.model_key = model_key
        
        try:
            if local_model_path and os.path.exists(local_model_path):
                # 로컬 모델 로드
                print(f"로컬 모델 로드: {local_model_path}")
                self.processor = YolosImageProcessor.from_pretrained(local_model_path)
                self.model = YolosForObjectDetection.from_pretrained(local_model_path)
                self.model_framework = "transformers"
            else:
                # 선택된 모델 정보 가져오기
                model_info = self.AVAILABLE_MODELS[model_key]
                model_name = model_info["name"]
                processor_type = model_info["processor_type"]
                framework = model_info["framework"]
                self.model_framework = framework
                
                # 온라인에서 모델 다운로드/로드
                print(f"선택된 모델: {model_key}")
                print(f"모델 설명: {model_info['description']}")
                print(f"프레임워크: {framework}")
                print(f"프로세서 타입: {processor_type}")
                
                if model_info.get("direct_download", False):
                    print(f"직접 다운로드 URI: {model_info['download_uri']}")
                else:
                    print(f"HuggingFace에서 모델 다운로드: {model_name}")
                    print(f"다운로드 URI: {model_info['download_uri']}")
                
                if token:
                    print("HuggingFace 토큰을 사용하여 인증합니다.")
                
                # 프레임워크별 모델 로드
                if framework == "ultralytics":
                    try:
                        # YOLOv5/YOLOv8 모델 처리
                        self.model = self._load_yolo_model(model_info, model_key)
                        print("⚠️  주의: YOLO 모델은 현재 버전에서 제한적으로 지원됩니다.")
                        print("완전한 지원을 위해서는 transformers 기반 모델을 사용하세요.")
                    except ImportError as e:
                        print(f"YOLO 라이브러리 로드 실패: {e}")
                        print("ultralytics 설치 필요: pip install ultralytics")
                        print("대신 transformers 기반 모델을 사용하세요: -m yolos-small")
                        raise
                
                elif framework == "transformers":
                    try:
                        if processor_type == "DetrImageProcessor":
                            if token:
                                self.processor = DetrImageProcessor.from_pretrained(model_name, token=token)
                                self.model = DetrForObjectDetection.from_pretrained(model_name, token=token)
                            else:
                                self.processor = DetrImageProcessor.from_pretrained(model_name)
                                self.model = DetrForObjectDetection.from_pretrained(model_name)
                        
                        elif processor_type == "YolosImageProcessor":
                            if token:
                                self.processor = YolosImageProcessor.from_pretrained(model_name, token=token)
                                self.model = YolosForObjectDetection.from_pretrained(model_name, token=token)
                            else:
                                self.processor = YolosImageProcessor.from_pretrained(model_name)
                                self.model = YolosForObjectDetection.from_pretrained(model_name)
                        
                        else:
                            raise ValueError(f"지원하지 않는 프로세서 타입: {processor_type}")
                    except ImportError as e:
                        print(f"Transformers 라이브러리 로드 실패: {e}")
                        print("transformers 설치 필요: pip install transformers")
                        raise
                
                else:
                    raise ValueError(f"지원하지 않는 프레임워크: {framework}")
            
            # Transformers 모델만 디바이스로 이동
            if hasattr(self, 'model') and hasattr(self.model, 'to') and self.model_framework == "transformers":
                self.model = self.model.to(self.device)
                print(f"모델이 {self.device}로 이동되었습니다.")
            
            print("🎉 모델 초기화 완료!")
            print(f"✅ 최종 사용 모델: {self.model_key}")
            if self.model_key in self.AVAILABLE_MODELS:
                final_info = self.AVAILABLE_MODELS[self.model_key]
                print(f"📋 모델 설명: {final_info['description']}")
                print(f"⚡ 성능: {final_info['performance']}")
            
        except Exception as e:
            print(f"\n❌ 모델 초기화 실패: {e}")
            print("\n해결 방법:")
            print("1. 필수 라이브러리 설치:")
            print("   pip install transformers huggingface-hub ultralytics torch torchvision")
            print("2. 인터넷 연결 확인")
            print("3. 다른 모델 시도: python license_plate_labeler.py --list-models")
            raise

    def _setup_device(self, force_cpu=False):
        """
        사용할 디바이스 설정 (GPU/CPU)
        
        Args:
            force_cpu (bool): CPU 사용 강제
            
        Returns:
            torch.device: 사용할 디바이스
        """
        if force_cpu:
            print("CPU 사용이 강제로 설정되었습니다.")
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            print(f"CUDA 사용 가능: {gpu_count}개의 GPU 감지")
            print(f"현재 GPU: {gpu_name} (디바이스 {current_gpu})")
            
            # GPU 메모리 정보 출력
            memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
            memory_cached = torch.cuda.memory_reserved(current_gpu) / 1024**3
            print(f"GPU 메모리 사용량: {memory_allocated:.2f}GB 할당됨, {memory_cached:.2f}GB 예약됨")
            
            return torch.device(f"cuda:{current_gpu}")
        else:
            print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            print("GPU 사용을 위해서는 다음을 확인하세요:")
            print("1. NVIDIA GPU가 설치되어 있는지")
            print("2. CUDA가 설치되어 있는지")
            print("3. PyTorch가 CUDA 지원으로 설치되어 있는지")
            return torch.device("cpu")
    
    def get_optimal_size(self, image_width, image_height, max_longest_edge=800, min_longest_edge=400):
        """
        이미지 크기에 따라 최적의 처리 크기 계산
        
        Args:
            image_width (int): 원본 이미지 너비
            image_height (int): 원본 이미지 높이
            max_longest_edge (int): 최대 긴 변 길이
            min_longest_edge (int): 최소 긴 변 길이
            
        Returns:
            dict: 최적화된 size 설정
        """
        longest_edge = max(image_width, image_height)
        shortest_edge = min(image_width, image_height)
        
        # 이미지가 너무 큰 경우 크기 줄이기
        if longest_edge > max_longest_edge:
            scale_factor = max_longest_edge / longest_edge
            optimal_longest_edge = max_longest_edge
            optimal_shortest_edge = int(shortest_edge * scale_factor)
        # 이미지가 너무 작은 경우 크기 늘리기
        elif longest_edge < min_longest_edge:
            scale_factor = min_longest_edge / longest_edge
            optimal_longest_edge = min_longest_edge
            optimal_shortest_edge = int(shortest_edge * scale_factor)
        else:
            # 적절한 크기로 8의 배수로 조정 (모델 효율성을 위해)
            optimal_longest_edge = ((longest_edge // 8) * 8)
            optimal_shortest_edge = ((shortest_edge // 8) * 8)
        
        return {
            "longest_edge": optimal_longest_edge,
            "shortest_edge": optimal_shortest_edge
        }

    def detect_license_plates_yolo(self, image_path, confidence_threshold=0.5):
        """
        YOLO 모델을 사용한 번호판 탐지
        
        Args:
            image_path (str): 이미지 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            tuple: (detections, original_size)
        """
        try:
            # 이미지 로드
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_size = image.size  # (width, height)
            
            print(f"원본 크기: {original_size}")
            
            # 모델 타입에 따라 다른 방식으로 추론
            model_info = self.AVAILABLE_MODELS[self.model_key]
            processor_type = model_info["processor_type"]
            
            if processor_type == "YOLOv5":
                # YOLOv5의 경우 모델 속성으로 confidence threshold 설정
                original_conf = getattr(self.model, 'conf', 0.25)  # 원래 값 백업
                self.model.conf = confidence_threshold
                
                try:
                    results = self.model(image_path)
                    
                    detections = []
                    # YOLOv5 결과 처리
                    if hasattr(results, 'pandas'):
                        try:
                            df = results.pandas().xyxy[0]
                            for _, row in df.iterrows():
                                detection = {
                                    'confidence': round(float(row['confidence']), 3),
                                    'label': int(row['class']),
                                    'bbox': [float(row['xmin']), float(row['ymin']), 
                                            float(row['xmax']), float(row['ymax'])]
                                }
                                detections.append(detection)
                        except Exception as pandas_error:
                            print(f"pandas 접근 실패: {pandas_error}, 대안 방법 시도")
                            if hasattr(results, 'pred') and len(results.pred) > 0:
                                pred = results.pred[0]
                                for detection in pred:
                                    x1, y1, x2, y2, conf, cls = detection.tolist()
                                    if conf >= confidence_threshold:
                                        detection_dict = {
                                            'confidence': round(conf, 3),
                                            'label': int(cls),
                                            'bbox': [x1, y1, x2, y2]
                                        }
                                        detections.append(detection_dict)
                    else:
                        if hasattr(results, 'pred') and len(results.pred) > 0:
                            pred = results.pred[0]
                            for detection in pred:
                                x1, y1, x2, y2, conf, cls = detection.tolist()
                                if conf >= confidence_threshold:
                                    detection_dict = {
                                        'confidence': round(conf, 3),
                                        'label': int(cls),
                                        'bbox': [x1, y1, x2, y2]
                                    }
                                    detections.append(detection_dict)
                finally:
                    # 원래 confidence 값으로 복원
                    self.model.conf = original_conf
                
            elif processor_type == "YOLOv8":
                try:
                    results = self.model(image_path, conf=confidence_threshold)
                except TypeError:
                    results = self.model(image_path)

                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf.item())
                            if conf >= confidence_threshold:
                                detection = {
                                    'confidence': round(conf, 3),
                                    'label': int(box.cls.item()),
                                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                                }
                                detections.append(detection)

            elif processor_type == "YOLOv11":
                try:
                    results = self.model(image_path, conf=confidence_threshold)
                except TypeError:
                    results = self.model(image_path)

                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf.item())
                            if conf >= confidence_threshold:
                                detection = {
                                    'confidence': round(conf, 3),
                                    'label': int(box.cls.item()),
                                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                                }
                                detections.append(detection)
            
            return detections, original_size
            
        except Exception as e:
            print(f"YOLO 모델 추론 실패: {e}")
            raise
        finally:
            # 메모리 정리
            if 'results' in locals():
                del results
            if 'detections' in locals():
                del detections
            torch.cuda.empty_cache()

    def detect_license_plates_transformers(self, image_path, confidence_threshold=0.5):
        """
        Transformers 모델을 사용한 번호판 탐지
        
        Args:
            image_path (str): 이미지 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            tuple: (detections, original_size)
        """
        try:
            # 이미지 로드
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                original_size = image.size  # (width, height)
            
            # 최적 처리 크기 계산
            optimal_size = self.get_optimal_size(original_size[0], original_size[1])
            
            print(f"원본 크기: {original_size}, 처리 크기: {optimal_size}")
            
            # 모델 입력 준비
            try:
                inputs = self.processor(
                    images=image, 
                    size=optimal_size, 
                    return_tensors="pt"
                )
            except (TypeError, ValueError) as e:
                print(f"새로운 size 파라미터 오류: {e}")
                print("기본 방식을 사용합니다.")
                try:
                    inputs = self.processor(images=image, return_tensors="pt")
                except Exception as fallback_error:
                    print(f"기본 방식도 실패: {fallback_error}")
                    height = optimal_size["shortest_edge"] if original_size[1] < original_size[0] else optimal_size["longest_edge"]
                    width = optimal_size["longest_edge"] if original_size[0] > original_size[1] else optimal_size["shortest_edge"]
                    inputs = self.processor(
                        images=image, 
                        size={"height": height, "width": width}, 
                        return_tensors="pt"
                    )
            
            # 입력 텐서를 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론 실행
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU 메모리 부족: {e}")
                    print("더 작은 이미지 크기로 다시 시도하거나 --max-size를 줄여보세요.")
                    print("또는 CPU 모드로 실행하려면 --force-cpu 옵션을 사용하세요.")
                    raise
                else:
                    raise
            
            # 결과 후처리를 위해 CPU로 이동
            target_sizes = torch.tensor([original_size[::-1]])  # (height, width)
            
            # outputs를 CPU로 이동하여 후처리
            if isinstance(outputs, dict):
                outputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
            else:
                outputs_cpu = {
                    'logits': outputs.logits.cpu(),
                    'pred_boxes': outputs.pred_boxes.cpu()
                }
            
            # 후처리 실행
            try:
                results = self.processor.post_process_object_detection(
                    outputs_cpu, target_sizes=target_sizes, threshold=confidence_threshold
                )[0]
            except Exception as e:
                print(f"후처리 오류: {e}")
                print("outputs 구조:", type(outputs))
                if isinstance(outputs, dict):
                    print("outputs keys:", list(outputs.keys()))
                
                # 수동으로 결과 처리
                try:
                    if isinstance(outputs, dict):
                        logits = outputs['logits'].cpu()
                        pred_boxes = outputs['pred_boxes'].cpu()
                    else:
                        logits = outputs.logits.cpu()
                        pred_boxes = outputs.pred_boxes.cpu()
                    
                    # 간단한 후처리
                    probs = torch.nn.functional.softmax(logits, -1)
                    scores, labels = probs[..., :-1].max(-1)
                    
                    # confidence threshold 적용
                    keep = scores > confidence_threshold
                    scores = scores[keep]
                    labels = labels[keep]
                    boxes = pred_boxes[keep]
                    
                    # 좌표 변환
                    img_w, img_h = target_sizes[0][1], target_sizes[0][0]
                    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                    boxes = boxes * scale_fct
                    
                    # cxcywh를 xyxy로 변환
                    boxes[:, :2] -= boxes[:, 2:] / 2
                    boxes[:, 2:] += boxes[:, :2]
                    
                    results = {
                        'scores': scores,
                        'labels': labels,
                        'boxes': boxes
                    }
                    
                except Exception as manual_error:
                    print(f"수동 처리도 실패: {manual_error}")
                    raise
            
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detection = {
                    'confidence': round(score.item(), 3),
                    'label': label.item(),
                    'bbox': box
                }
                detections.append(detection)
            
            return detections, original_size
            
        except Exception as e:
            print(f"Transformers 모델 추론 실패: {e}")
            raise
        finally:
            # 메모리 정리
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            if 'outputs_cpu' in locals():
                del outputs_cpu
            if 'results' in locals():
                del results
            torch.cuda.empty_cache()

    def detect_license_plates(self, image_path, confidence_threshold=0.5):
        """
        이미지에서 번호판 탐지 (모델 프레임워크에 따라 적절한 방법 선택)
        
        Args:
            image_path (str): 이미지 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            tuple: (detections, original_size) 탐지된 번호판의 바운딩 박스 정보와 원본 이미지 크기
        """
        if self.model_framework == "ultralytics":
            return self.detect_license_plates_yolo(image_path, confidence_threshold)
        elif self.model_framework == "transformers":
            return self.detect_license_plates_transformers(image_path, confidence_threshold)
        else:
            raise ValueError(f"지원하지 않는 모델 프레임워크: {self.model_framework}")
    
    def convert_to_yolo_format(self, bbox, image_width, image_height):
        """
        바운딩 박스를 YOLO 형식으로 변환
        
        Args:
            bbox (list): [x_min, y_min, x_max, y_max] 형식의 바운딩 박스
            image_width (int): 이미지 너비
            image_height (int): 이미지 높이
            
        Returns:
            tuple: (x_center, y_center, width, height) - 모두 정규화된 값
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 중심점 계산
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        
        # 너비, 높이 계산
        width = x_max - x_min
        height = y_max - y_min
        
        # 정규화 (0~1 범위)
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        return x_center, y_center, width, height
    
    def save_yolo_label(self, detections, image_size, output_path, class_id=0):
        """
        YOLO 형식의 라벨 파일 저장
        
        Args:
            detections (list): 탐지 결과 리스트
            image_size (tuple): (width, height)
            output_path (str): 출력 파일 경로
            class_id (int): 클래스 ID (번호판은 보통 0)
        """
        image_width, image_height = image_size
        
        with open(output_path, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                x_center, y_center, width, height = self.convert_to_yolo_format(
                    bbox, image_width, image_height
                )
                
                # YOLO 형식: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def process_single_image(self, image_path, output_dir, confidence_threshold=0.5, 
                           save_visualization=True, undetected_dir=None):
        """
        단일 이미지 처리 (사이즈 정보 출력 포함)
        
        Args:
            image_path (str): 입력 이미지 경로
            output_dir (str): 출력 디렉토리
            confidence_threshold (float): 신뢰도 임계값
            save_visualization (bool): 시각화 결과 저장 여부
            undetected_dir (str): 탐지되지 않은 이미지 저장 디렉토리 (None이면 저장 안함)
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 및 확장자 추출
        image_path_obj = Path(image_path)
        image_name = image_path_obj.stem
        image_ext = image_path_obj.suffix
        
        print(f"처리 중: {image_path}")
        
        # 이미지 크기 정보 출력
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"이미지 크기: {width}x{height}")
        
        # 번호판 탐지
        detections, image_size = self.detect_license_plates(image_path, confidence_threshold)
        
        if len(detections) == 0:
            print(f"번호판이 탐지되지 않았습니다: {image_path}")
            
            # 탐지되지 않은 이미지 저장 (지정된 디렉토리가 있을 때만)
            if undetected_dir:
                os.makedirs(undetected_dir, exist_ok=True)
                undetected_path = os.path.join(undetected_dir, f"{image_name}_undetected{image_ext}")
                try:
                    shutil.copy2(image_path, undetected_path)
                    print(f"탐지되지 않은 이미지 저장: {undetected_path}")
                except Exception as e:
                    print(f"이미지 복사 실패: {e}")
            return
        
        # 원본 이미지를 output 디렉토리에 복사
        output_image_path = os.path.join(output_dir, f"{image_name}{image_ext}")
        try:
            shutil.copy2(image_path, output_image_path)
            print(f"원본 이미지 복사: {output_image_path}")
        except Exception as e:
            print(f"원본 이미지 복사 실패: {e}")
        
        # YOLO 라벨 파일 저장 (원본 이미지명과 동일하게)
        label_path = os.path.join(output_dir, f"{image_name}.txt")
        self.save_yolo_label(detections, image_size, label_path)
        print(f"라벨 파일 저장: {label_path}")
        
        # 시각화 결과 저장 (선택사항)
        if save_visualization:
            vis_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
            self.visualize_detections(image_path, detections, vis_path)

        print(f"탐지된 번호판 수: {len(detections)}")
        for i, detection in enumerate(detections):
            print(f"  번호판 {i+1}: 신뢰도 {detection['confidence']:.3f}")

        # 번호판 영역 크롭 이미지 저장
        try:
            with Image.open(image_path) as img:
                for idx, detection in enumerate(detections, 1):
                    x_min, y_min, x_max, y_max = map(int, detection['bbox'])
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img.width, x_max)
                    y_max = min(img.height, y_max)
                    cropped = img.crop((x_min, y_min, x_max, y_max))
                    crop_filename = f"{image_name}_plate{idx:03d}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    cropped.save(crop_path)
                    print(f"크롭된 번호판 저장: {crop_path}")
        except Exception as e:
            print(f"번호판 크롭 저장 실패: {e}")
    
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
    
    def process_directory(self, input_dir, output_dir, confidence_threshold=0.5,
                         save_visualization=True, undetected_dir=None, image_extensions=None):
        """
        디렉토리 내 모든 이미지 처리
        
        Args:
            input_dir (str): 입력 디렉토리
            output_dir (str): 출력 디렉토리
            confidence_threshold (float): 신뢰도 임계값
            save_visualization (bool): 시각화 결과 저장 여부
            undetected_dir (str): 탐지되지 않은 이미지 저장 디렉토리 (None이면 저장 안함)
            image_extensions (list): 처리할 이미지 확장자 리스트
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_dir)
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"입력 디렉토리에서 이미지 파일을 찾을 수 없습니다: {input_dir}")
            return
        
        print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            self.process_single_image(
                str(image_file), output_dir, confidence_threshold, save_visualization, undetected_dir
            )

def main():
    try:
        # 모델 키 목록을 동적으로 가져오기
        available_model_keys = list(LicensePlateYOLOLabeler.AVAILABLE_MODELS.keys())
        
        parser = argparse.ArgumentParser(
            description="번호판 탐지 및 YOLO 라벨 생성기",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
사용 예시:
  1. 기본 사용:
     python %(prog)s -i image.jpg -o output_dir
  
  2. 특정 모델 선택:
     python %(prog)s -i image.jpg -o output_dir -m yolos-small
  
  3. 신뢰도 조정:
     python %(prog)s -i input_dir -o output_dir -c 0.7
  
  4. 시각화 비활성화:
     python %(prog)s -i input_dir -o output_dir --no-viz
  
  5. 미탐지 이미지 저장:
     python %(prog)s -i input_dir -o output_dir -e undetected_dir
  
  6. CPU 강제 사용:
     python %(prog)s -i input_dir -o output_dir --force-cpu

모델 카테고리:
  🔥 최고 정확도: yolov11x (mAP@50: 0.9813)
  💪 고성능: yolov11l, yolov11m, detr-resnet50
  ⚖️ 균형잡힌: yolov11s, yolos-small, yolos-rego
  🚀 경량: yolov11n, yolov5m, yolov8s

HuggingFace 토큰 설정:
  1. 환경 변수로 설정:       export HF_TOKEN='your_token_here'
  2. 명령줄 인자로 설정:     python %(prog)s -i input_dir -o output_dir -t 'your_token_here'

필수 라이브러리 설치:
  pip install transformers huggingface-hub ultralytics torch torchvision opencv-python
            """
        )
        
        parser.add_argument("--input", "-i", 
                           help="입력 이미지 파일 또는 디렉토리 경로")
        parser.add_argument("--output", "-o",
                           help="출력 디렉토리 경로")
        parser.add_argument("--model", "-m", type=str, default="yolos-small",
                           choices=available_model_keys,
                           help=f"사용할 모델 선택 (기본값: yolos-small)\n"
                                f"🔥 최고 정확도:\n"
                                f"  - yolov11x: YOLOv11x 모델, 최고 정확도 (mAP@50: 0.9813, 114MB)\n"
                                f"💪 고성능:\n"
                                f"  - yolov11l: YOLOv11 large 모델 (51.2MB)\n"
                                f"  - yolov11m: YOLOv11 medium 모델 (40.5MB)\n"
                                f"  - detr-resnet50: DETR + ResNet50 (160MB)\n"
                                f"⚖️ 균형잡힌:\n"
                                f"  - yolov11s: YOLOv11 small 모델 (19.2MB)\n"
                                f"  - yolos-small: YOLO + Vision Transformer (90MB)\n"
                                f"  - yolos-rego: YOLOS + 차량+번호판 동시 탐지 (90MB)\n"
                                f"🚀 경량:\n"
                                f"  - yolov11n: YOLOv11 nano 모델 (5.47MB)\n"
                                f"  - yolov5m: YOLOv5 medium 모델 (40MB)\n"
                                f"  - yolov8s: YOLOv8 small 모델 (22MB)")
        parser.add_argument("--token", "-t", type=str,
                           help="HuggingFace 액세스 토큰 (private 모델 접근시 필요)\n"
                                "토큰은 https://huggingface.co/settings/tokens 에서 생성 가능\n"
                                "환경 변수 HF_TOKEN으로도 설정 가능")
        parser.add_argument("--list-models", action="store_true",
                           help="사용 가능한 모델 목록과 상세 정보 출력")
        parser.add_argument("--local-model", type=str,
                           help="로컬 모델 경로 (오프라인 사용시)\n"
                                "HuggingFace 모델을 로컬에 다운로드하여 사용할 때 지정")
        parser.add_argument("--confidence", "-c", type=float, default=0.5,
                           help="신뢰도 임계값 (0.0-1.0, 기본값: 0.5)\n"
                                "높은 값: 더 확실한 탐지만 허용\n"
                                "낮은 값: 더 많은 후보 탐지 허용")
        parser.add_argument("--no-viz", action="store_true",
                           help="시각화 결과 저장 안함\n"
                                "탐지된 번호판을 표시한 이미지 생성하지 않음")
        parser.add_argument("--undetected-dir", "-e", type=str,
                           help="탐지되지 않은 이미지를 저장할 디렉토리 경로\n"
                                "번호판이 탐지되지 않은 이미지를 별도로 저장")
        parser.add_argument("--max-size", type=int, default=800,
                           help="처리할 최대 이미지 크기 (longest edge, 기본값: 800)\n"
                                "큰 이미지는 이 크기로 축소되어 처리됨\n"
                                "메모리 사용량과 처리 속도에 영향")
        parser.add_argument("--force-cpu", action="store_true",
                           help="GPU 사용을 비활성화하고 CPU만 사용\n"
                                "GPU 메모리 부족시 또는 호환성 문제시 사용")
        
        args = parser.parse_args()
        
        # 모델 목록 출력 요청시
        if args.list_models:
            LicensePlateYOLOLabeler.list_available_models()
            return
        
        # input과 output이 필수인지 확인 (--list-models가 아닌 경우에만)
        if not args.input or not args.output:
            parser.error("--input과 --output 인수가 필요합니다. (--list-models 사용시 제외)")
        
        # 입력 경로 존재 확인
        if not os.path.exists(args.input):
            parser.error(f"입력 경로가 존재하지 않습니다: {args.input}")
        
        # 출력 디렉토리 생성
        os.makedirs(args.output, exist_ok=True)
        
        # 모델 키 유효성 검사 (로컬 모델이 아닌 경우)
        if not args.local_model and args.model not in available_model_keys:
            print(f"오류: 지원하지 않는 모델 키 '{args.model}'")
            print(f"사용 가능한 모델: {', '.join(available_model_keys[:10])}...")
            print("전체 모델 목록: python license_plate_labeler.py --list-models")
            return
        
        # 라벨러 초기화
        try:
            print("\n=== 번호판 탐지 YOLO 라벨링 생성기 ===")
            print(f"선택된 모델: {args.model}")
            print(f"입력 경로: {args.input}")
            print(f"출력 경로: {args.output}")
            print(f"신뢰도 임계값: {args.confidence}")
            print(f"CPU 강제 사용: {args.force_cpu}")
            
            labeler = LicensePlateYOLOLabeler(
                model_key=args.model, 
                local_model_path=args.local_model, 
                force_cpu=args.force_cpu,
                token=args.token
            )
        except Exception as e:
            print(f"\n❌ 모델 초기화 실패: {e}")
            print("\n해결 방법:")
            print("1. 필수 라이브러리 설치: pip install transformers huggingface-hub ultralytics")
            print("2. 인터넷 연결 확인")
            print("3. 다른 모델 시도해보세요 (--model 옵션)")
            return
        
        # 사용자가 지정한 최대 크기 적용
        if hasattr(args, 'max_size'):
            original_get_optimal_size = labeler.get_optimal_size
            def custom_get_optimal_size(width, height, max_longest_edge=args.max_size, min_longest_edge=400):
                return original_get_optimal_size(width, height, max_longest_edge, min_longest_edge)
            labeler.get_optimal_size = custom_get_optimal_size
        
        # 입력이 파일인지 디렉토리인지 확인
        input_path = Path(args.input)
        
        try:
            if input_path.is_file():
                # 단일 파일 처리
                print(f"\n단일 이미지 처리 중: {args.input}")
                labeler.process_single_image(
                    args.input, args.output, args.confidence, not args.no_viz, args.undetected_dir
                )
            elif input_path.is_dir():
                # 디렉토리 처리
                print(f"\n디렉토리 처리 중: {args.input}")
                labeler.process_directory(
                    args.input, args.output, args.confidence, not args.no_viz, args.undetected_dir
                )
            else:
                print(f"입력 경로가 유효하지 않습니다: {args.input}")
                return
                
            print("\n✅ 처리가 완료되었습니다!")
            print(f"결과가 저장된 경로: {args.output}")
            
        except Exception as e:
            print(f"\n❌ 처리 중 오류 발생: {e}")
            print("\n해결 방법:")
            print("1. 입력 이미지가 올바른 형식인지 확인")
            print("2. 신뢰도 임계값을 조정해보세요 (--confidence 옵션)")
            print("3. 다른 모델을 시도해보세요 (--model 옵션)")
            return
            
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        print("\n프로그램을 종료합니다.")
        return

if __name__ == "__main__":
    main()