import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("NO")
    exit()

while True:
    ret, frame = cap.read()
    
    if cv.waitKey(1) == ord("n") or cv.waitKey(1) == ord("N"):
        custom_color = cv.COLOR_BGR2BGRA
    
    if cv.waitKey(1) == ord("o") or cv.waitKey(1) == ord("O"):
        custom_color = cv.COLOR_
    
    
    if not ret:
        print("exit")
        break

    color = cv.cvtColor(frame, custom_color)
    cv.imshow("CUSTOM CAMERA", color)
    if cv.waitKey(1) == ord("q") or cv.waitKey(1) == ord("Q"):
        break

cap.release()
cv.destroyAllWindows()