import numpy as np
import cv2
import os
import sys
from sys import argv

directory = "input/"
csvMap = {
    "Australia-Training.csv": "australia",
    "Canada-Training.csv": "canada",
    "China-Training.csv": "china",
    "Germany-Training.csv": "germany",
    "Japan-Training.csv": "japan",
    "Korea-Training.csv": "korea",
    "Singapore-Training.csv": "singapore",
    "SwissPost-Training.csv": "switzerland",
    "Unknown-Training.csv": "unknown",
    "USA-Training.csv": "usa"
}


def readCSV(filename, country):

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()[1:]  # Starts reading from second line
    except Exception as e:
        print("\n\nFile not found")
        print("Please place your file in the same directory or provide an absolute path")
        print("In the event you're using data.csv, please place it in the same directory as this program file")
        exit(0)

    loopIndex = 0
    for loopIndex in range(len(lines)):
        currLine = lines[loopIndex]
        tokens = currLine.strip().split(",")
        fileName = tokens[0]
        skewImage(fileName, country)


def skewImage(img, country):
    image = cv2.imread(directory + img, cv2.IMREAD_UNCHANGED)

    scale_percent = 30  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    kernel = np.ones((5, 5), np.uint8)
    resized = cv2.erode(resized, kernel, iterations=1)
    resized = cv2.dilate(resized, kernel, iterations=1)

    gray = cv2.bitwise_not(resized)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    coords = np.column_stack(np.where(thresh == 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

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
    # cv2.imshow("Input " + img + "  " + country, resized)
    # cv2.imshow("Rotated " + img + "  " + country, rotated)

    status = cv2.imwrite(directory + "/align/" + country + "/" + img.split(".")[0] + "_align.jpg", rotated)
    cv2.waitKey(0)


def main():
    os.makedirs(directory + "/align", exist_ok=True)
    for (csv, country) in csvMap.items():
        os.makedirs(directory + "/align/" + country, exist_ok=True)
        readCSV("extra_docs/" + csv, country)
        print("done with " + country)


if __name__ == '__main__':
    main()
