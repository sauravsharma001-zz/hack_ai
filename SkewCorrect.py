# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image file")
# args = vars(ap.parse_args())

# load the image from disk
image = cv2.imread("atos.jpg", cv2.IMREAD_UNCHANGED)

scale_percent = 30  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
kernel = np.ones((5,5), np.uint8)
resized = cv2.erode(resized, kernel, iterations=1)
resized = cv2.dilate(resized, kernel, iterations=1)



# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(resized)
#
# # threshold the image, setting all foreground pixels to
# # 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# thresh = cv2.dilate(thresh, kernel, iterations=1)
thresh = cv2.erode(thresh, kernel, iterations=1)
thresh = cv2.dilate(thresh, kernel, iterations=1)


#
# # grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh == 0))
angle = cv2.minAreaRect(coords)[-1]

print(coords)
print(angle)

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)
    print(angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = -angle
    print(angle)

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it

rotated = cv2.resize(rotated, dim, interpolation=cv2.INTER_AREA)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", resized)

cv2.imshow("Rotated", rotated)
cv2.waitKey(0)