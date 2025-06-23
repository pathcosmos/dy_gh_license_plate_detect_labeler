from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 한글 지원
result = ocr.ocr('cropped_plate.jpg')

for line in result[0]:
    for word_info in line:
        text = word_info[1][0]
        score = word_info[1][1]
        print(f"문자: {text}, 신뢰도: {score}")