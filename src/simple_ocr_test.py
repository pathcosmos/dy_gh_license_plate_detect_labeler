#!/usr/bin/env python3
"""
간단한 OCR 테스트 스크립트
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

def test_ocr():
    """OCR 테스트"""
    print("=== OCR 테스트 시작 ===")
    
    # 이미지 경로
    image_path = "./temp_data/out_source_plate/20180713-182905-000537-0_plate001.jpg"
    
    if not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않습니다: {image_path}")
        return
    
    print(f"1. 이미지 로딩: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("이미지 로딩 실패!")
        return
    
    print(f"이미지 크기: {image.shape}")
    
    print("2. PaddleOCR 초기화...")
    try:
        ocr = PaddleOCR(lang='korean')
        print("PaddleOCR 초기화 성공")
    except Exception as e:
        print(f"PaddleOCR 초기화 실패: {e}")
        return
    
    print("3. OCR 추론 시작...")
    try:
        result = ocr.predict(image)
        print(f"OCR 결과: {result}")
        print(f"결과 타입: {type(result)}")
        print(f"결과 길이: {len(result) if result else 0}")
        
        if result:
            print("결과 상세 분석:")
            for i, item in enumerate(result):
                print(f"  [{i}]: {item}")
                if isinstance(item, list):
                    for j, sub_item in enumerate(item):
                        print(f"    [{i}][{j}]: {sub_item}")
        else:
            print("OCR 결과가 비어있습니다.")
            
    except Exception as e:
        print(f"OCR 추론 실패: {e}")
        import traceback
        print(f"스택 트레이스: {traceback.format_exc()}")

if __name__ == "__main__":
    test_ocr() 