#!/bin/bash

# 현재 스크립트의 디렉토리로 이동
cd "$(dirname "$0")"

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT="$(cd .. && pwd)"

# Python 경로에 프로젝트 루트 추가
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 가상환경 활성화 (있는 경우)
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
fi

# 테스트 디렉토리 생성
mkdir -p test_data/input
mkdir -p test_data/output
mkdir -p test_data/undetected
mkdir -p logs

# 테스트 실행
echo "Running tests..."
python -m unittest discover -s . -p "test_*.py" -v

# 테스트 결과 저장
TEST_RESULT=$?

# 테스트 실패 시 로그 파일 확인
if [ $TEST_RESULT -ne 0 ]; then
    echo -e "\n=== 테스트 실패 로그 확인 ==="
    ls -t logs/test_results_*.log | head -n 1 | xargs cat
fi

# 가상환경 비활성화 (활성화된 경우)
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

exit $TEST_RESULT 