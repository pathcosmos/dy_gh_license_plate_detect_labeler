import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
import logging
from paddleocr import PaddleOCR
import re


class LicensePlateOCR:
    """
    번호판 이미지에서 텍스트를 인식하는 클래스
    PaddleOCR을 사용하여 한국어 번호판 텍스트를 추출합니다.
    """
    
    def __init__(self, use_textline_orientation=True, lang='korean'):
        """
        LicensePlateOCR 초기화
        
        Args:
            use_textline_orientation (bool): 텍스트 방향 분류 사용 여부
            lang (str): OCR 언어 설정 ('korean', 'en', 'ch' 등)
        """
        self.use_textline_orientation = use_textline_orientation
        self.lang = lang
        
        # PaddleOCR 초기화 (번호판 인식에 최적화된 설정)
        try:
            self.ocr = PaddleOCR(
                use_textline_orientation=use_textline_orientation,
                lang=lang,
                # 번호판 인식을 위한 추가 설정 (최신 파라미터명 사용)
                text_det_thresh=0.1,  # 텍스트 검출 임계값 낮춤
                text_det_box_thresh=0.1,  # 박스 검출 임계값 낮춤
                # 이미지 전처리 설정
                text_det_limit_side_len=960,  # 이미지 크기 제한
                text_det_limit_type='min'  # 최소 크기로 제한
            )
            logging.info(f"PaddleOCR 초기화 완료 (언어: {lang})")
        except Exception as e:
            logging.error(f"PaddleOCR 초기화 실패: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        이미지 전처리
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 이진화
            _, binary = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            return binary
            
        except Exception as e:
            logging.error(f"이미지 전처리 실패: {e}")
            # 전처리 실패 시 원본 이미지 반환
            return cv2.imread(image_path)
    
    def extract_license_plate_text(
        self, image_path: str, confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        번호판 이미지에서 텍스트 추출
        
        Args:
            image_path (str): 번호판 이미지 경로
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            List[Dict]: 추출된 텍스트 정보 리스트
        """
        try:
            print(f"=== 텍스트 추출 시작: {image_path} ===")
            logging.info(f"=== 텍스트 추출 시작: {image_path} ===")
            
            # 원본 이미지로 먼저 시도
            print("1. 이미지 로딩 중...")
            logging.info("1. 이미지 로딩 중...")
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            print(f"이미지 로드 성공: {image_path}, 크기: {original_image.shape}")
            logging.info(
                f"이미지 로드 성공: {image_path}, "
                f"크기: {original_image.shape}"
            )
            
            # OCR 수행 (원본 이미지 사용)
            print("2. PaddleOCR 모델 초기화 및 추론 중...")
            print("   (첫 번째 실행 시 모델 다운로드로 시간이 오래 걸릴 수 있습니다)")
            logging.info("2. PaddleOCR 모델 초기화 및 추론 중...")
            logging.info("   (첫 번째 실행 시 모델 다운로드로 시간이 오래 걸릴 수 있습니다)")
            
            try:
                result = self.ocr.predict(original_image)
                print(f"OCR 원본 결과: {result}")
                logging.info(f"OCR 원본 결과: {result}")
                print(f"OCR 결과 타입: {type(result)}")
                logging.info(f"OCR 결과 타입: {type(result)}")
                
                if result is None:
                    print("OCR 결과가 None입니다!")
                    logging.error("OCR 결과가 None입니다!")
                    return []
                
                result_len = len(result) if result else 0
                print(f"OCR 추론 완료: 결과 타입: {type(result)}, 길이: {result_len}")
                logging.info(
                    f"OCR 추론 완료: 결과 타입: {type(result)}, 길이: {result_len}"
                )
                
                # 새로운 PaddleOCR API 결과 구조 파싱
                extracted_texts = []
                print("3. 새로운 API 결과 구조 파싱 중...")
                logging.info("3. 새로운 API 결과 구조 파싱 중...")
                
                if result and len(result) > 0:
                    first_result = result[0]
                    print(f"첫 번째 결과 키들: {list(first_result.keys())}")
                    logging.info(f"첫 번째 결과 키들: {list(first_result.keys())}")
                    
                    # rec_texts와 rec_scores 추출
                    if 'rec_texts' in first_result and 'rec_scores' in first_result:
                        rec_texts = first_result['rec_texts']
                        rec_scores = first_result['rec_scores']
                        rec_polys = first_result.get('rec_polys', [])
                        
                        print(f"인식된 텍스트: {rec_texts}")
                        print(f"신뢰도 점수: {rec_scores}")
                        logging.info(f"인식된 텍스트: {rec_texts}")
                        logging.info(f"신뢰도 점수: {rec_scores}")
                        
                        # 각 텍스트와 신뢰도 매칭
                        for i, (text, confidence) in enumerate(zip(rec_texts, rec_scores)):
                            print(f"텍스트 {i}: '{text}', 신뢰도: {confidence}")
                            logging.info(f"텍스트 {i}: '{text}', 신뢰도: {confidence}")
                            
                            # 신뢰도 임계값 확인
                            if confidence >= confidence_threshold:
                                # 바운딩 박스 정보 추출
                                bbox = rec_polys[i] if i < len(rec_polys) else []
                                
                                # 번호판 패턴 확인
                                is_plate = self._is_valid_license_plate(text)
                                
                                extracted_texts.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'is_license_plate': is_plate
                                })
                                
                                if is_plate:
                                    print(f"    ✓ 번호판 텍스트 확인: {text} (신뢰도: {confidence:.3f})")
                                    logging.info(
                                        f"    ✓ 번호판 텍스트 확인: {text} "
                                        f"(신뢰도: {confidence:.3f})"
                                    )
                                else:
                                    print(f"    - 일반 텍스트: {text} (신뢰도: {confidence:.3f})")
                                    logging.info(
                                        f"    - 일반 텍스트: {text} "
                                        f"(신뢰도: {confidence:.3f})"
                                    )
                            else:
                                print(f"    ✗ 신뢰도 낮음으로 무시: {text} (신뢰도: {confidence:.3f} < {confidence_threshold})")
                                logging.info(
                                    f"    ✗ 신뢰도 낮음으로 무시: {text} "
                                    f"(신뢰도: {confidence:.3f} < "
                                    f"{confidence_threshold})"
                                )
                    else:
                        print("rec_texts 또는 rec_scores를 찾을 수 없습니다.")
                        logging.warning("rec_texts 또는 rec_scores를 찾을 수 없습니다.")
                
            except Exception as ocr_error:
                print(f"PaddleOCR 추론 중 오류 발생: {ocr_error}")
                logging.error(f"PaddleOCR 추론 중 오류 발생: {ocr_error}")
                print(f"오류 타입: {type(ocr_error)}")
                logging.error(f"오류 타입: {type(ocr_error)}")
                import traceback
                print(f"스택 트레이스: {traceback.format_exc()}")
                logging.error(f"스택 트레이스: {traceback.format_exc()}")
                return []
            
            if not extracted_texts:
                print(f"텍스트를 찾을 수 없습니다: {image_path}")
                logging.warning(f"텍스트를 찾을 수 없습니다: {image_path}")
                print("=== 텍스트 추출 완료: 결과 없음 ===")
                logging.info("=== 텍스트 추출 완료: 결과 없음 ===")
                return []
            
            print(f"4. 최종 추출된 텍스트 수: {len(extracted_texts)}")
            logging.info(f"4. 최종 추출된 텍스트 수: {len(extracted_texts)}")
            
            # 번호판 텍스트 개수 계산
            plate_count = sum(
                1 for t in extracted_texts if t['is_license_plate']
            )
            print(f"   - 번호판 텍스트: {plate_count}개")
            print(f"   - 일반 텍스트: {len(extracted_texts) - plate_count}개")
            logging.info(
                f"   - 번호판 텍스트: {plate_count}개"
            )
            logging.info(
                f"   - 일반 텍스트: {len(extracted_texts) - plate_count}개"
            )
            
            if extracted_texts:
                print("추출된 텍스트 목록:")
                logging.info("추출된 텍스트 목록:")
                for i, text_info in enumerate(extracted_texts, 1):
                    plate_mark = "✓" if text_info['is_license_plate'] else "-"
                    print(f"   {i}. {plate_mark} '{text_info['text']}' (신뢰도: {text_info['confidence']:.3f})")
                    logging.info(
                        f"   {i}. {plate_mark} '{text_info['text']}' "
                        f"(신뢰도: {text_info['confidence']:.3f})"
                    )
            
            print("=== 텍스트 추출 완료 ===")
            logging.info("=== 텍스트 추출 완료 ===")
            return extracted_texts
            
        except Exception as e:
            print(f"텍스트 추출 실패: {e}")
            logging.error(f"텍스트 추출 실패: {e}")
            print("=== 텍스트 추출 실패 ===")
            logging.error("=== 텍스트 추출 실패 ===")
            import traceback
            print(f"스택 트레이스: {traceback.format_exc()}")
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            return []
    
    def _is_valid_license_plate(self, text: str) -> bool:
        """
        추출된 텍스트가 유효한 번호판 패턴인지 확인
        
        Args:
            text (str): 검증할 텍스트
            
        Returns:
            bool: 유효한 번호판 패턴 여부
        """
        # 공백 제거 및 대문자 변환
        text = text.replace(' ', '').upper()
        
        # 한국 번호판 패턴들 (더 유연하게)
        patterns = [
            # 일반 번호판: 12가3456, 123가4567
            r'^\d{2,3}[가-힣]\d{4}$',
            # 지역명 번호판: 서울12가3456, 경기123가4567
            r'^[가-힣]{2,3}\d{2,3}[가-힣]\d{4}$',
            # 특수 번호판: 12-3456, 123-4567
            r'^\d{2,3}-\d{4}$',
            # 외교관 번호판: 12-3456
            r'^\d{2}-\d{4}$',
            # 영업용 번호판: 12가3456
            r'^\d{2}[가-힣]\d{4}$',
            # 임시 번호판: 12-3456
            r'^\d{2}-\d{4}$',
            # 숫자만 (일부 케이스)
            r'^\d{6,8}$',
            # 영문+숫자 조합
            r'^[A-Z]{1,3}\d{4,6}$',
            r'^\d{2,4}[A-Z]{1,3}\d{2,4}$',
            # 더 유연한 패턴들
            r'^\d{1,4}[가-힣A-Z]\d{1,4}$',  # 숫자+문자+숫자
            r'^[가-힣A-Z]\d{1,4}[가-힣A-Z]\d{1,4}$',  # 문자+숫자+문자+숫자
            r'^\d{1,4}$',  # 숫자만 (짧은 경우)
            r'^[가-힣A-Z]{1,4}$',  # 문자만 (짧은 경우)
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def process_single_image(
        self, image_path: str, output_dir: str = None,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        단일 이미지 처리
        
        Args:
            image_path (str): 처리할 이미지 경로
            output_dir (str): 결과 저장 디렉토리
            confidence_threshold (float): 신뢰도 임계값
            
        Returns:
            Dict: 처리 결과
        """
        try:
            # 텍스트 추출
            extracted_texts = self.extract_license_plate_text(
                image_path, confidence_threshold
            )
            
            result = {
                'image_path': image_path,
                'extracted_texts': extracted_texts,
                'success': len(extracted_texts) > 0
            }
            
            # 결과 저장
            if output_dir:
                self._save_ocr_result(result, output_dir)
            
            return result
            
        except Exception as e:
            logging.error(f"이미지 처리 실패: {e}")
            return {
                'image_path': image_path,
                'extracted_texts': [],
                'success': False,
                'error': str(e)
            }
    
    def _save_ocr_result(self, result: Dict, output_dir: str):
        """
        OCR 결과를 파일로 저장
        
        Args:
            result (Dict): OCR 결과
            output_dir (str): 저장할 디렉토리
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 이미지 파일명에서 확장자 제거
            image_name = Path(result['image_path']).stem
            
            # OCR 결과 텍스트 파일 저장
            ocr_result_path = os.path.join(
                output_dir, f"{image_name}_ocr_result.txt"
            )
            
            with open(ocr_result_path, 'w', encoding='utf-8') as f:
                f.write(f"이미지: {result['image_path']}\n")
                f.write(f"처리 성공: {result['success']}\n")
                f.write(f"추출된 텍스트 수: {len(result['extracted_texts'])}\n")
                f.write("-" * 50 + "\n")
                
                if result['extracted_texts']:
                    for i, text_info in enumerate(
                        result['extracted_texts'], 1
                    ):
                        f.write(f"텍스트 {i}: {text_info['text']}\n")
                        f.write(f"신뢰도: {text_info['confidence']:.3f}\n")
                        f.write(f"번호판 여부: {text_info['is_license_plate']}\n")
                        f.write(f"좌표: {text_info['bbox']}\n")
                        f.write("-" * 30 + "\n")
                else:
                    f.write("추출된 텍스트가 없습니다.\n")
            
            logging.info(f"OCR 결과 저장: {ocr_result_path}")
            
        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
    
    def process_directory(
        self, input_dir: str, output_dir: str = None,
        confidence_threshold: float = 0.5,
        image_extensions: List[str] = None
    ) -> List[Dict]:
        """
        디렉토리 내 모든 이미지 처리
        
        Args:
            input_dir (str): 입력 디렉토리
            output_dir (str): 출력 디렉토리
            confidence_threshold (float): 신뢰도 임계값
            image_extensions (List[str]): 처리할 이미지 확장자 리스트
            
        Returns:
            List[Dict]: 모든 처리 결과
        """
        logging.info(f"=== 디렉토리 일괄 처리 시작: {input_dir} ===")
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        results = []
        image_files = []
        
        logging.info("1. 이미지 파일 검색 중...")
        # 번호판 이미지 파일 찾기 (*_plate*.jpg 패턴)
        for ext in image_extensions:
            # *_plate*.jpg 패턴으로 번호판 이미지 찾기
            plate_pattern = f"*_plate*{ext}"
            plate_files = list(Path(input_dir).glob(plate_pattern))
            plate_files.extend(Path(input_dir).glob(plate_pattern.upper()))
            
            if plate_files:
                logging.info(f"   번호판 이미지 ({ext}): {len(plate_files)}개 발견")
                image_files.extend(plate_files)
            
            # 일반 이미지 파일도 찾기 (기존 기능 유지)
            general_pattern = f"*{ext}"
            general_files = list(Path(input_dir).glob(general_pattern))
            general_files.extend(Path(input_dir).glob(general_pattern.upper()))
            
            # 중복 제거
            for file in general_files:
                if file not in image_files:
                    image_files.append(file)
        
        # 파일명으로 정렬
        image_files.sort()
        
        logging.info(f"2. 총 처리할 이미지 파일 수: {len(image_files)}")
        logging.info("처리할 파일 목록:")
        for i, file in enumerate(image_files[:10]):  # 처음 10개만 출력
            logging.info(f"   {i+1}. {file.name}")
        if len(image_files) > 10:
            logging.info(
                f"   ... 및 {len(image_files) - 10}개 더"
            )
        
        # 각 이미지 처리
        logging.info("3. 이미지별 텍스트 추출 시작...")
        successful_count = 0
        failed_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            logging.info(
                f"\n--- 처리 중 ({i}/{len(image_files)}): {image_file.name} ---"
            )
            result = self.process_single_image(
                str(image_file),
                output_dir,
                confidence_threshold
            )
            results.append(result)
            
            if result['success']:
                successful_count += 1
                logging.info(f"✓ 성공: {image_file.name}")
            else:
                failed_count += 1
                logging.info(f"✗ 실패: {image_file.name}")
        
        # 통계 출력
        logging.info("\n=== 처리 완료 ===")
        logging.info(f"총 처리 파일: {len(results)}개")
        logging.info(f"성공: {successful_count}개")
        logging.info(f"실패: {failed_count}개")
        logging.info(f"성공률: {successful_count/len(results)*100:.1f}%")
        
        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='번호판 OCR 처리')
    parser.add_argument(
        '--input', '-i', required=True,
        help='입력 이미지 파일 또는 디렉토리 경로'
    )
    parser.add_argument(
        '--output', '-o', default='./ocr_results',
        help='결과 저장 디렉토리 (기본값: ./ocr_results)'
    )
    parser.add_argument(
        '--confidence', '-c', type=float, default=0.5,
        help='신뢰도 임계값 (기본값: 0.5)'
    )
    parser.add_argument(
        '--lang', '-l', default='korean',
        help='OCR 언어 설정 (기본값: korean)'
    )
    parser.add_argument(
        '--use-textline-orientation', action='store_true', default=True,
        help='텍스트 방향 분류 사용 (기본값: True)'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG,  # DEBUG 레벨로 변경하여 모든 로그 출력
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
            logging.FileHandler('ocr_debug.log', encoding='utf-8')  # 파일 저장
        ]
    )
    
    # 즉시 출력을 위한 print 함수
    def debug_print(message):
        print(f"[DEBUG] {message}")
        logging.debug(message)
    
    try:
        # OCR 객체 생성
        ocr = LicensePlateOCR(
            use_textline_orientation=args.use_textline_orientation,
            lang=args.lang
        )
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 단일 파일 처리
            result = ocr.process_single_image(
                str(input_path),
                args.output,
                args.confidence
            )
            print(f"처리 완료: {result['success']}")
            
        elif input_path.is_dir():
            # 디렉토리 처리
            results = ocr.process_directory(
                str(input_path),
                args.output,
                args.confidence
            )
            successful_count = sum(1 for r in results if r['success'])
            print(f"처리 완료: {len(results)}개 중 {successful_count}개 성공")
            
        else:
            print(f"입력 경로가 유효하지 않습니다: {args.input}")
            
    except Exception as e:
        logging.error(f"처리 중 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
