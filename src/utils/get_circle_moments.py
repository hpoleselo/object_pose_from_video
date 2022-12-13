import cv2
import numpy as np
import os

"""

https://www.pythonpool.com/opencv-moments/
https://en.wikipedia.org/wiki/Image_moment
"""

# Image path to be system-agnostic
img_path = os.path.join('../test_data', 'circle.jpg')

# Reading image as binary right away
# Or use just a (img, 0) instead of passing cv2.IMREAD_GRAYSCALE
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Applying binary thresholding
ret, thresh = cv2.threshold(img, 127, 255, 0)

moments = cv2.moments(thresh)

print(f"Image size: {img.shape}")
print(f"Moments: {moments}")

# Drawing centroid and contours from the moments
# Area for binary images is denoted by M00
# Centroid is denoted as M10/M00 and M01/M00.
x = int(moments['m10'] / moments['m00'])
y = int(moments['m01'] / moments['m00'])


cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
 
cv2.putText(img, "Centroid", (x - 25, y - 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Detects edges from the image, 30 and 200 are aperture sizes
canny_edges = cv2.Canny(img, 30, 200)

# Extracts contours from the image
contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow("Edge", canny_edges)
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.waitKey(0)

# Image Orientation Can be Dervied by using the second order central moments
# TO construct a covariance matrix:
