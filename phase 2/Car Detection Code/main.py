import cv2
import numpy as np
import os
os.chdir(R"C:\Users\habib\PycharmProjects\pythonProject1")

w1 = 170
h1 = 170
offset = 5
y1 = 650

delay = 60

detec = []
carros = 0

def pega_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture("Road1.mp4")

BGS = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=100, detectShadows=True)

while True:
    ret, frame1 = cap.read()

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(grey, (9, 9), 5)
    img_sub = BGS.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))

    contor, h = cv2.findContours(dilat, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (300, y1), (1100, y1), (176, 130, 39), 2)

    for(i, c) in enumerate(contor):
        (x, y, w, h) = cv2.boundingRect(c)

        validar_contorno = (40<w<150) and (40<h<150)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        center = pega_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), 1)

    cv2.imshow("video original", frame1)
    cv2.imshow("dilat", dilat)
    cv2.imshow("img_sub", img_sub)

    if cv2.waitKey(1) == 'q':
        break

cv2.destroyAllWindows()
cap.release()






