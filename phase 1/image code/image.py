import cv2
import numpy as np
import matplotlib.pylab as plt

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
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=30)
    img = cv2.addWeighted(img, 1.25, blank_image, 3, 2)
    return img

img = cv2.imread("lines4.jpg.jpeg")
print(img.shape)
height = img.shape[0]
width = img.shape[1]
region_of_interest_vertices = [(0, height), (width/2, height/2), (width, height)]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 200 , 500)
cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32), )
lines = cv2.HoughLinesP(cropped_image,rho=6,theta=np.pi/180,threshold=400,lines=np.array([]),
                        minLineLength=200,maxLineGap=200)

image_with_lines = drow_the_lines(img, lines)
plt.imshow(image_with_lines)
plt.show()
