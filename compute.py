import cv2
import sys
import numpy as np
import math

def illumination(image, alpha):
    rows,cols,channels = image.shape
    print image.shape
    newImage = np.zeros((rows,cols,1), np.uint8)
    for x in xrange (rows):
        for y in xrange (cols):
            if image[x][y][0] == 0:
                image[x][y][0] = 1
            if image[x][y][1] == 0:
                image[x][y][1] = 1
            if image[x][y][2] == 0:
                image[x][y][2] = 1
            newValue = 0.5 + math.log(image[x][y][1]) - alpha*math.log(image[x][y][2]) - (1-alpha)*math.log(image[x][y][0])
            newImage[x][y] = newValue * 255
    return newImage

def illuminationHSV(image):
    rows,cols,channels = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.split(image_hsv)
    return hsv[2]

windowName = "Original Image"
windowName2 = "New Image"

if (len(sys.argv) == 2):

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)

    lower_threshold = 25
    upper_threshold = 120
    smoothing_neighbourhood = 3
    sobel_size = 3

    img = cv2.imread(sys.argv[1])

    smoothing_neighbourhood = max(3, smoothing_neighbourhood)
    if not(smoothing_neighbourhood % 2):
        smoothing_neighbourhood = smoothing_neighbourhood + 1

    sobel_size = max(3, sobel_size)
    if not(sobel_size % 2):
        sobel_size = sobel_size + 1

    gray_frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    gray_frame2 = illuminationHSV(img)

    # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())

    smoothed1 = cv2.GaussianBlur(gray_frame1, (smoothing_neighbourhood, smoothing_neighbourhood), 0)
    smoothed2 = cv2.GaussianBlur(gray_frame2, (smoothing_neighbourhood, smoothing_neighbourhood), 0)

    # perform canny edge detection

    canny1 = cv2.Canny(smoothed1, lower_threshold, upper_threshold, apertureSize=sobel_size)
    canny2 = cv2.Canny(smoothed2, lower_threshold, upper_threshold, apertureSize=sobel_size)

    #cv2.imshow(windowName, img)
    cv2.imshow("Normal canny", canny1)
    cv2.moveWindow("Normal canny", 0, 100)
    cv2.imshow("HSV Value canny", canny2)
    cv2.moveWindow("HSV Value canny", 640, 100)

    cv2.waitKey()

else:
    print("No image file specified.")
