import cv2

# =========================================
# VIDEO
# =========================================

VIDEO_PATH = "video.mp4"

# =========================================
# OPEN VIDEO
# =========================================

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Ошибка чтения видео")
    exit()

# =========================================
# FRAME INFO
# =========================================

height, width, _ = frame.shape

print("================================")
print(f"Frame width: {width}")
print(f"Frame height: {height}")
print("================================")

# =========================================
# SPLIT LEFT / RIGHT
# =========================================

half_width = width // 2

left_frame = frame[:, :half_width]
right_frame = frame[:, half_width:]

# =========================================
# SHOW
# =========================================

cv2.imshow("FULL FRAME", frame)

cv2.imshow("LEFT CAMERA", left_frame)

cv2.imshow("RIGHT CAMERA", right_frame)

cv2.waitKey(0)

cv2.destroyAllWindows()

cap.release()