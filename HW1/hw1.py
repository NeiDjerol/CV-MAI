import cv2
import sys

squares = []
square_size = 10

def show_clicked(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: 
        # Добавляем квадрат с центром в точке клика
        squares.append((x, y, square_size))
        print(f"Квадрат добавлен в ({x}, {y})")
    

script_name = sys.argv[0]
args = str(sys.argv[1])  # Кладем в args путь для видео, как аргумент командной строки
print (f"args : {args}")

capture = cv2.VideoCapture(str(args))

if not capture.isOpened():
    print("Ошибка: Не удалось открыть видеофайл")
    exit()

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', show_clicked)

while(True):
    ret, frame = capture.read()

    if not ret: 
        print("Видео закончилось")
        exit()

    # Our operations on the frame come here
    for (x, y, size) in squares:
        half = size // 2
        # Рисуем квадрат (зеленый, толщина 2)
        cv2.rectangle(frame, 
                    (x - half, y - half), 
                    (x + half, y + half), 
                    (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        squares.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()