import cv2 as cv
import random as r

delay = 1
random = 0
random_stop = True
custom_color = cv.COLOR_BGR2BGRA
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("NO")
    exit()

while True:
    ret, frame = cap.read()

    if cv.waitKey(delay) == ord("n") or cv.waitKey(delay) == ord("N"):
        random_stop = True
        random = 0
        custom_color = cv.COLOR_BGR2BGRA
    
    if cv.waitKey(delay) == ord("r") or cv.waitKey(delay) == ord("R"):
        random_stop = True
        random = r.randint(1, 5)

    if cv.waitKey(delay) == ord("c") or cv.waitKey(delay) == ord("C"):
        random_stop = False

    if random_stop == False:
        random = r.randint(1, 5)
    
    if random == 1:
        custom_color = cv.COLOR_RGB2YUV
    
    if random == 2:
        custom_color = cv.COLOR_RGB2BGR

    if random == 3:
        custom_color = cv.COLOR_RGB2GRAY

    if random == 4:
        custom_color = cv.COLOR_RGB2HLS

    if random == 5:
        custom_color = cv.COLOR_RGB2HSV

    if not ret:
        print("exit")
        break

    default_color = custom_color
    
    color = cv.cvtColor(frame, default_color)
    
    cv.imshow("CUSTOM CAMERA", color)

    if cv.waitKey(delay) == ord("q") or cv.waitKey(delay) == ord("Q"):
        break

cap.release()
cv.destroyAllWindows()