import os
import cv2
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import shutil

class LicensePlateYOLOLabeler:
    def __init__(self, model_name="nickmuchi/yolos-small-finetuned-license-plate-detection", local_model_path=None):
        """
        번호판 탐지 및 YOLO 라벨링 파일 생성기 초기화
        
        Args:
            model_name (str): 사용할 HuggingFace 모델명
            local_model_path (str): 로컬 모델 경로 (오프라인 사용시)
        """
        print("모델 로딩 중...")
        
        try:
            if local_model_path and os.path.exists(local_model_path):
                # 로컬 모델 로드
                print(f"로컬 모델 로드: {local_model_path}")
                self.processor = YolosImageProcessor.from_pretrained(local_model_path)
                self.model = YolosForObjectDetection.from_pretrained(local_model_path)
            else:
                # 온라인에서 모델 다운로드/로드
                print(f"HuggingFace에서 모델 다운로드: {model_name}")
                self.processor = YolosImageProcessor.from_pretrained(model_name)
                self.model = YolosForObjectDetection.from_pretrained(model_name)
            
            print("모델 로딩 완료!")
            
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            print("인터넷 연결을 확인하거나 로컬 모델 경로를 지정해주세요.")
            raise
    
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
    
    def detect_license_plates(self, image_path, confidence_threshold=0.5):
        """
        이미지에서 번호판 탐지 (사이즈 최적화 포함)
        
        Args:
            image_path (str): 이미지 파일 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            list: 탐지된 번호판의 바운딩 박스 정보
        """
        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        # 최적 처리 크기 계산
        optimal_size = self.get_optimal_size(original_size[0], original_size[1])
        
        print(f"원본 크기: {original_size}, 처리 크기: {optimal_size}")
        
        # 모델 입력 준비 (올바른 size 형식 사용)
        try:
            # 새로운 방식: size parameter 사용 (shortest_edge와 longest_edge 모두 포함)
            inputs = self.processor(
                images=image, 
                size=optimal_size, 
                return_tensors="pt"
            )
        except (TypeError, ValueError) as e:
            # 호환성을 위한 fallback
            print(f"새로운 size 파라미터 오류: {e}")
            print("기본 방식을 사용합니다.")
            try:
                # 구버전 호환성을 위한 시도
                inputs = self.processor(images=image, return_tensors="pt")
            except Exception as fallback_error:
                print(f"기본 방식도 실패: {fallback_error}")
                # 마지막 시도: 명시적으로 height, width 설정
                height = optimal_size["shortest_edge"] if original_size[1] < original_size[0] else optimal_size["longest_edge"]
                width = optimal_size["longest_edge"] if original_size[0] > original_size[1] else optimal_size["shortest_edge"]
                inputs = self.processor(
                    images=image, 
                    size={"height": height, "width": width}, 
                    return_tensors="pt"
                )
        
        # 추론 실행
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([original_size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection = {
                'confidence': round(score.item(), 3),
                'label': label.item(),
                'bbox': box  # [x_min, y_min, x_max, y_max]
            }
            detections.append(detection)
            
        return detections, original_size
    
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
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)
    
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
        
        # 파일명 추출
        image_name = Path(image_path).stem
        
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
                undetected_path = os.path.join(undetected_dir, f"{image_name}_undetected.jpg")
                try:
                    shutil.copy2(image_path, undetected_path)
                    print(f"탐지되지 않은 이미지 저장: {undetected_path}")
                except Exception as e:
                    print(f"이미지 복사 실패: {e}")
            return
        
        # YOLO 라벨 파일 저장
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
    parser = argparse.ArgumentParser(description="번호판 탐지 및 YOLO 라벨 생성")
    parser.add_argument("--input", "-i", required=True, 
                       help="입력 이미지 파일 또는 디렉토리 경로")
    parser.add_argument("--output", "-o", required=True,
                       help="출력 디렉토리 경로")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="신뢰도 임계값 (기본값: 0.5)")
    parser.add_argument("--no-viz", action="store_true",
                       help="시각화 결과 저장 안함")
    parser.add_argument("--undetected-dir", "-e", type=str,
                       help="탐지되지 않은 이미지를 저장할 디렉토리 경로")
    parser.add_argument("--max-size", type=int, default=800,
                       help="처리할 최대 이미지 크기 (longest edge, 기본값: 800)")
    
    args = parser.parse_args()
    
    # 라벨러 초기화
    labeler = LicensePlateYOLOLabeler()
    
    # 사용자가 지정한 최대 크기 적용
    if hasattr(args, 'max_size'):
        original_get_optimal_size = labeler.get_optimal_size
        def custom_get_optimal_size(width, height, max_longest_edge=args.max_size, min_longest_edge=400):
            return original_get_optimal_size(width, height, max_longest_edge, min_longest_edge)
        labeler.get_optimal_size = custom_get_optimal_size
    
    # 입력이 파일인지 디렉토리인지 확인
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 단일 파일 처리
        labeler.process_single_image(
            args.input, args.output, args.confidence, not args.no_viz, args.undetected_dir
        )
    elif input_path.is_dir():
        # 디렉토리 처리
        labeler.process_directory(
            args.input, args.output, args.confidence, not args.no_viz, args.undetected_dir
        )
    else:
        print(f"입력 경로가 유효하지 않습니다: {args.input}")

if __name__ == "__main__":
    # 예제 사용법
    print("=== 번호판 탐지 YOLO 라벨링 생성기 ===")
    print("\n사용법:")
    print("1. 단일 이미지: python script.py -i image.jpg -o output_dir")
    print("2. 디렉토리: python script.py -i input_dir -o output_dir")
    print("3. 신뢰도 조정: python script.py -i input_dir -o output_dir -c 0.7")
    print("4. 시각화 없이: python script.py -i input_dir -o output_dir --no-viz")
    print("5. 최대 크기 설정: python script.py -i input_dir -o output_dir --max-size 1024")
    print("6. 미탐지 이미지 저장: python script.py -i input_dir -o output_dir -e undetected_dir")
    
    # CLI 모드로 실행
    main()