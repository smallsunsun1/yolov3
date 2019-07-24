import numpy as np
import cv2


image = cv2.imread("./6.jpg")
h, w = np.shape(image)[:2]
# x0 = int(w * 0.6155555555555555)
# y0 = int(h * 0.5621301775147929)
# x1 = int(w * 0.8822222222222222)
# y1 = int(h * 0.8609467455621301)

x0 = int(w * 0.5621301775147929)
y0 = int(h * 0.6155555555555555)
x1 = int(w * 0.8609467455621301)
y1 = int(h * 0.8822222222222222)
image = cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
cv2.imshow("win", image)
cv2.waitKey()
