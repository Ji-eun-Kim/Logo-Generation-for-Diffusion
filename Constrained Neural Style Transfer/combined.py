import cv2
import numpy as np

logo = 'k'
background = '' #'_background' # ''

original_path = f'/home/work/rvc/logo/input/contents/logo/{logo}/{logo}_origin.png'
transferred_path = f'/home/work/rvc/logo/output/{logo}_logo{background}_vs_flower/5000.jpg'
result_path = f'/home/work/rvc/logo/combined/{logo}/{logo}_logo{background}.png'



# 이미지 로드
original = cv2.imread(original_path)
transferred = cv2.imread(transferred_path,  cv2.IMREAD_UNCHANGED)


# 트랜스퍼된 이미지를 원본 이미지 크기에 맞게 조정
transferred = cv2.resize(transferred, (original.shape[1], original.shape[0]))

# 그레이스케일로 변환
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
transferred_gray = cv2.cvtColor(transferred, cv2.COLOR_BGR2GRAY)

# 원본 이미지에서 트랜스퍼된 부분의 마스크 생성
mask = cv2.threshold(transferred_gray, 250, 255, cv2.THRESH_BINARY_INV)[1]

# 마스크를 이용해 원본 이미지와 트랜스퍼된 이미지의 차이 계산
diff = cv2.absdiff(original_gray, transferred_gray)
diff = cv2.bitwise_and(diff, diff, mask=mask)

# 차이가 있는 부분만 마스크로 생성
change_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

# 마스크 확장 (선택적)
kernel = np.ones((3,3), np.uint8)
change_mask = cv2.dilate(change_mask, kernel, iterations=1)

# 변경된 부분만 선택
changed_part = cv2.bitwise_and(transferred, transferred, mask=change_mask)

# 원본 이미지에서 변경될 부분 제거
original_background = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(change_mask))

# 변경된 부분과 원본 배경 합성
result = cv2.add(original_background, changed_part)

# 결과 저장

cv2.imwrite(result_path, result)