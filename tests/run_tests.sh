#!/bin/bash

# 테스트 디렉토리로 이동
cd "$(dirname "$0")"

# Python 가상환경 활성화 (있는 경우)
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# 테스트 실행
python -m unittest test_license_plate_labeler.py -v

# 테스트 결과 확인
if [ $? -eq 0 ]; then
    echo "모든 테스트가 성공적으로 완료되었습니다."
else
    echo "일부 테스트가 실패했습니다."
    exit 1
fi 