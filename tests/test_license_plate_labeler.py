import unittest
import os
import sys
import time
from pathlib import Path
import shutil
import logging
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import traceback
import requests

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.license_plate_labeler import LicensePlateYOLOLabeler

class TestLicensePlateLabeler(unittest.TestCase):
    # 클래스 변수로 HF_TOKEN 설정
    HF_TOKEN = None  # 기본값은 None
    
    @classmethod
    def setUpClass(cls):
        """테스트 환경 설정"""
        # 테스트용 디렉토리 설정
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.input_dir = cls.test_dir / "input"
        cls.output_dir = cls.test_dir / "output"
        cls.undetected_dir = cls.test_dir / "undetected"
        
        # 로깅 설정 (먼저 로거 초기화)
        cls._setup_logging()
        
        # HF_TOKEN 설정 확인 및 입력
        cls._setup_hf_token()
        
        # 실패한 모델 추적을 위한 리스트
        cls.failed_models = []
        
        # 입력 이미지 확인
        cls.input_images = list(cls.input_dir.glob("*.jpg")) + list(cls.input_dir.glob("*.png"))
        if not cls.input_images:
            raise ValueError(f"입력 이미지가 없습니다: {cls.input_dir}")
        cls.logger.info(f"테스트할 입력 이미지 {len(cls.input_images)}개 발견")
    
    @classmethod
    def _setup_logging(cls):
        """로깅 설정"""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"test_results_{timestamp}.log"
        
        # 로거가 이미 존재하는 경우 핸들러 제거
        if hasattr(cls, 'logger'):
            for handler in cls.logger.handlers[:]:
                cls.logger.removeHandler(handler)
        
        # 로거 설정
        cls.logger = logging.getLogger(__name__)
        cls.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        cls.logger.addHandler(file_handler)
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        cls.logger.addHandler(console_handler)
        
        cls.logger.info(f"로깅 설정 완료: {log_file}")
    
    @classmethod
    def _setup_hf_token(cls):
        """HF_TOKEN 설정 확인 및 입력"""
        # 환경 변수에서 HF_TOKEN 확인
        if os.getenv('HF_TOKEN'):
            cls.logger.info("HF_TOKEN이 환경 변수에서 설정되었습니다.")
            return
        
        # 사용자에게 HF_TOKEN 입력 요청
        print("\n" + "="*50)
        print("HF_TOKEN이 설정되지 않았습니다.")
        print("Hugging Face 모델을 다운로드하기 위해 HF_TOKEN이 필요합니다.")
        print("1. https://huggingface.co/settings/tokens 에서 토큰을 생성하세요.")
        print("2. 아래에 토큰을 입력하세요.")
        print("="*50 + "\n")
        
        while True:
            try:
                token = input("HF_TOKEN을 입력하세요: ").strip()
                if not token:
                    print("토큰이 입력되지 않았습니다. 다시 시도하세요.")
                    continue
                
                # 토큰 형식 검증 (기본적인 형식만 확인)
                if not token.startswith("hf_"):
                    print("올바른 HF_TOKEN 형식이 아닙니다. 'hf_'로 시작해야 합니다.")
                    continue
                
                # 토큰 설정
                os.environ['HF_TOKEN'] = token
                cls.logger.info("HF_TOKEN이 성공적으로 설정되었습니다.")
                break
                
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                sys.exit(1)
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")
                print("다시 시도하세요.")
    
    @classmethod
    def _verify_hf_token(cls, token: str) -> bool:
        """HF_TOKEN 유효성 검증"""
        try:
            # 간단한 API 요청으로 토큰 검증
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers)
            return response.status_code == 200
        except Exception as e:
            cls.logger.error(f"토큰 검증 중 오류 발생: {e}")
            return False
    
    def setUp(self):
        """각 테스트 전 실행"""
        # 출력 디렉토리 초기화 (내용물만 삭제)
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 미탐지 디렉토리 초기화 (내용물만 삭제)
        if self.undetected_dir.exists():
            for item in self.undetected_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        self.undetected_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """각 테스트 후 실행"""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 실행"""
        # 실패한 모델이 있으면 로그에 기록
        if cls.failed_models:
            cls.logger.error("\n실패한 모델 목록:")
            for model in cls.failed_models:
                cls.logger.error(f"- {model}")
            
            # 실패 보고서 생성
            report_file = Path(__file__).parent / "logs" / f"failed_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("실패한 모델 목록:\n")
                for model in cls.failed_models:
                    f.write(f"- {model}\n")
    
    def _download_all_models(self):
        """모든 모델 다운로드"""
        self.logger.info("\n=== 모델 다운로드 단계 시작 ===")
        
        # 실제 사용 가능한 모델만 필터링
        all_models = [model_key for model_key in LicensePlateYOLOLabeler.AVAILABLE_MODELS.keys()]
        total_models = len(all_models)
        downloaded_models = []
        failed_downloads = {}
        
        self.logger.info(f"총 {total_models}개 모델 다운로드 시작...")
        self.logger.info(f"다운로드할 모델 목록: {', '.join(all_models)}")
        
        for i, model_name in enumerate(all_models, 1):
            try:
                self.logger.info(f"\n[{i}/{total_models}] {model_name} 모델 다운로드 중...")
                
                # 모델 정보 로깅
                model_info = LicensePlateYOLOLabeler.AVAILABLE_MODELS[model_name]
                self.logger.info(f"모델 정보: {model_info['description']}")
                self.logger.info(f"프레임워크: {model_info['framework']}")
                self.logger.info(f"크기: {model_info['size']}")
                
                # 모델 로드 (다운로드 포함)
                try:
                    self.logger.info(f"모델 다운로드/로드 시작: {model_name}")
                    labeler = LicensePlateYOLOLabeler(
                        model_key=model_name,
                        token=os.getenv('HF_TOKEN')
                    )
                    self.logger.info(f"✅ {model_name} 모델 다운로드 성공")
                    downloaded_models.append(model_name)
                except Exception as e:
                    self.logger.error(f"❌ {model_name} 모델 다운로드 실패")
                    self.logger.error(f"에러 메시지: {str(e)}")
                    self.logger.error(f"에러 타입: {type(e).__name__}")
                    self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
                    failed_downloads[model_name] = {
                        'error': str(e),
                        'stack_trace': sys.exc_info(),
                        'model_info': model_info,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    continue
                
            except Exception as e:
                self.logger.error(f"❌ {model_name} 모델 처리 중 예외 발생")
                self.logger.error(f"에러 메시지: {str(e)}")
                self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
                failed_downloads[model_name] = {
                    'error': str(e),
                    'stack_trace': sys.exc_info(),
                    'model_info': model_info if 'model_info' in locals() else None,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                continue
        
        # 다운로드 결과 보고
        self.logger.info("\n=== 모델 다운로드 결과 ===")
        self.logger.info(f"총 모델 수: {total_models}")
        self.logger.info(f"다운로드 성공: {len(downloaded_models)}")
        self.logger.info(f"다운로드 실패: {len(failed_downloads)}")
        
        if failed_downloads:
            self.logger.error("\n다운로드 실패한 모델:")
            for model, info in failed_downloads.items():
                self.logger.error(f"\n모델: {model}")
                self.logger.error(f"실패 시간: {info['timestamp']}")
                if info['model_info']:
                    self.logger.error(f"모델 정보: {info['model_info']['description']}")
                self.logger.error(f"에러 메시지: {info['error']}")
        
        return downloaded_models, failed_downloads

    def test_all_models(self):
        """모든 모델 테스트"""
        # HF_TOKEN 확인 및 설정
        if not os.getenv('HF_TOKEN'):
            self._setup_hf_token()
        
        # 토큰 유효성 검증
        token = os.getenv('HF_TOKEN')
        if not self._verify_hf_token(token):
            print("\n" + "="*50)
            print("현재 설정된 HF_TOKEN이 유효하지 않습니다.")
            print("새로운 토큰을 입력해주세요.")
            print("="*50 + "\n")
            self._setup_hf_token()
        
        # 1단계: 모든 모델 다운로드
        downloaded_models, failed_downloads = self._download_all_models()
        
        if not downloaded_models:
            self.logger.error("다운로드 성공한 모델이 없습니다. 테스트를 중단합니다.")
            return
        
        # 2단계: 다운로드 성공한 모델들만 테스트
        self.logger.info("\n=== 모델 테스트 단계 시작 ===")
        total_models = len(downloaded_models)
        successful_models = []
        failed_models_info = {}
        
        self.logger.info(f"총 {total_models}개 모델 테스트 시작...")
        self.logger.info(f"테스트할 모델 목록: {', '.join(downloaded_models)}")
        
        for i, model_name in enumerate(downloaded_models, 1):
            try:
                self.logger.info(f"\n[{i}/{total_models}] {model_name} 모델 테스트 중...")
                start_time = time.time()
                
                # 모델별 출력 디렉토리 설정
                model_output_dir = self.output_dir / model_name
                model_undetected_dir = self.undetected_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)
                model_undetected_dir.mkdir(parents=True, exist_ok=True)
                
                # 모델 정보 로깅
                model_info = LicensePlateYOLOLabeler.AVAILABLE_MODELS[model_name]
                self.logger.info(f"모델 정보: {model_info['description']}")
                self.logger.info(f"프레임워크: {model_info['framework']}")
                self.logger.info(f"크기: {model_info['size']}")
                
                # 모델 로드 (이미 다운로드되어 있으므로 빠르게 로드됨)
                try:
                    self.logger.info(f"모델 로드 시작: {model_name}")
                    labeler = LicensePlateYOLOLabeler(
                        model_key=model_name,
                        token=os.getenv('HF_TOKEN')
                    )
                    self.logger.info(f"모델 로드 성공: {model_name}")
                except Exception as e:
                    self.logger.error(f"모델 로드 실패: {model_name}")
                    self.logger.error(f"에러 메시지: {str(e)}")
                    self.logger.error(f"에러 타입: {type(e).__name__}")
                    self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
                    failed_models_info[model_name] = {
                        'error': str(e),
                        'stack_trace': sys.exc_info(),
                        'model_info': model_info,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.failed_models.append(model_name)
                    continue
                
                # 각 이미지 처리
                total_images = len(self.input_images)
                successful_images = 0
                failed_images = []
                
                for img_idx, img_path in enumerate(self.input_images, 1):
                    try:
                        self.logger.info(f"이미지 처리 중 [{img_idx}/{total_images}]: {img_path.name}")
                        
                        # 이미지 크기 및 형식 확인
                        with Image.open(img_path) as img:
                            self.logger.info(f"이미지 크기: {img.size}, 모드: {img.mode}")
                        
                        # 이미지 처리
                        result = labeler.process_single_image(
                            str(img_path),
                            str(model_output_dir),
                            confidence_threshold=0.1,
                            save_visualization=True,
                            undetected_dir=str(model_undetected_dir)
                        )
                        
                        # 결과 검증
                        if result is not None:
                            if len(result) > 0:
                                successful_images += 1
                                self.logger.info(f"이미지 처리 성공: {img_path.name}")
                                self.logger.info(f"탐지된 객체 수: {len(result)}")
                                for idx, detection in enumerate(result, 1):
                                    self.logger.info(f"  객체 {idx}: {detection}")
                            else:
                                self.logger.warning(f"이미지에서 객체를 찾지 못함: {img_path.name}")
                                failed_images.append(img_path.name)
                        else:
                            self.logger.error(f"이미지 처리 실패: {img_path.name}")
                            failed_images.append(img_path.name)
                            
                    except Exception as e:
                        self.logger.error(f"이미지 처리 실패: {img_path.name}")
                        self.logger.error(f"에러 메시지: {str(e)}")
                        self.logger.error(f"에러 타입: {type(e).__name__}")
                        self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
                        failed_images.append(img_path.name)
                        continue
                
                # 모델 테스트 결과 검증
                if successful_images > 0:
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    self.logger.info(f"✅ {model_name} 모델 테스트 성공")
                    self.logger.info(f"  - 처리된 이미지: {total_images}개")
                    self.logger.info(f"  - 성공한 이미지: {successful_images}개")
                    self.logger.info(f"  - 실패한 이미지: {len(failed_images)}개")
                    if failed_images:
                        self.logger.info(f"  - 실패한 이미지 목록: {', '.join(failed_images)}")
                    self.logger.info(f"  - 처리 시간: {processing_time:.2f}초")
                    successful_models.append(model_name)
                else:
                    self.logger.error(f"❌ {model_name} 모델이 어떤 이미지에서도 탐지를 수행하지 못했습니다.")
                    failed_models_info[model_name] = {
                        'error': "모든 이미지에서 탐지 실패",
                        'stack_trace': None,
                        'model_info': model_info,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.failed_models.append(model_name)
                
            except Exception as e:
                error_msg = str(e)
                stack_trace = sys.exc_info()
                self.logger.error(f"❌ {model_name} 모델 테스트 실패: {error_msg}")
                self.logger.error(f"스택 트레이스: {''.join(traceback.format_tb(stack_trace[2]))}")
                
                failed_models_info[model_name] = {
                    'error': error_msg,
                    'stack_trace': stack_trace,
                    'model_info': model_info if 'model_info' in locals() else None,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.failed_models.append(model_name)
                continue
        
        # 최종 결과 보고
        self.logger.info("\n=== 테스트 결과 요약 ===")
        self.logger.info(f"총 모델 수: {total_models}")
        self.logger.info(f"성공한 모델 수: {len(successful_models)}")
        self.logger.info(f"실패한 모델 수: {len(self.failed_models)}")
        
        if self.failed_models:
            self.logger.error("\n실패한 모델 상세 정보:")
            for model in self.failed_models:
                info = failed_models_info[model]
                self.logger.error(f"\n모델: {model}")
                self.logger.error(f"실패 시간: {info['timestamp']}")
                if info['model_info']:
                    self.logger.error(f"모델 정보: {info['model_info']['description']}")
                    self.logger.error(f"프레임워크: {info['model_info']['framework']}")
                    self.logger.error(f"크기: {info['model_info']['size']}")
                self.logger.error(f"에러 메시지: {info['error']}")
                if info['stack_trace']:
                    self.logger.error(f"스택 트레이스: {''.join(traceback.format_tb(info['stack_trace'][2]))}")
            
            # 실패 보고서 생성
            report_file = Path(__file__).parent / "logs" / f"failed_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== 실패한 모델 상세 보고서 ===\n\n")
                for model in self.failed_models:
                    info = failed_models_info[model]
                    f.write(f"모델: {model}\n")
                    f.write(f"실패 시간: {info['timestamp']}\n")
                    if info['model_info']:
                        f.write(f"모델 정보: {info['model_info']['description']}\n")
                        f.write(f"프레임워크: {info['model_info']['framework']}\n")
                        f.write(f"크기: {info['model_info']['size']}\n")
                    f.write(f"에러 메시지: {info['error']}\n")
                    if info['stack_trace']:
                        f.write(f"스택 트레이스:\n{''.join(traceback.format_tb(info['stack_trace'][2]))}\n")
                    f.write("\n" + "="*50 + "\n\n")
        
        # 실패한 모델이 있더라도 테스트는 계속 진행
        if self.failed_models:
            self.logger.warning(f"다음 모델들의 테스트가 실패했습니다: {', '.join(self.failed_models)}")
            self.logger.warning("실패한 모델에 대한 자세한 정보는 로그 파일을 참조하세요.")

if __name__ == '__main__':
    unittest.main()