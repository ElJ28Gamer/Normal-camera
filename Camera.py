import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("NO")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("exit")
        break

    color = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    cv.imshow("CUSTOM CAMERA", color)
    if cv.waitKey(1) == ord("q") or cv.waitKey(1) == ord("Q"):
        break

cap.release()
cv.destroyAllWindows()