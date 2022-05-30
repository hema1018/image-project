import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 250, 0), thickness=10)
    img = cv2.addWeighted(img, 1.25, blank_image, 3, 2)
    return img

def process(img):
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 200, 500)
    cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32), )
    lines = cv2.HoughLinesP(cropped_image, rho=6,theta=np.pi/180, threshold=200, lines=np.array([]),
                            minLineLength=50,maxLineGap=75)

    image_with_lines = drow_the_lines(img, lines)
    return image_with_lines

cap = cv2.VideoCapture("Road1.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow(" frame ", frame)
    if cv2.waitKey(75) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()