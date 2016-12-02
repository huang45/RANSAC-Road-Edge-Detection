import cv2
import sys
import numpy as np

def nothing(x):
    pass

windowName = "Original Image"
windowName2 = "Edges"
keep_processing = True

def illuminationHSV(image):
    rows,cols,channels = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(image_hsv)
    cv2.normalize(h, h, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(s, s, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    newImage = cv2.add(h,s)
    cv2.normalize(newImage, newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return newImage

if (len(sys.argv) == 2):

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)

    lower_threshold = 25
    cv2.createTrackbar("lower", windowName2, lower_threshold, 255, nothing)
    upper_threshold = 120
    cv2.createTrackbar("upper", windowName2, upper_threshold, 255, nothing)
    smoothing_neighbourhood = 3
    cv2.createTrackbar("smoothing", windowName2, smoothing_neighbourhood, 15, nothing)
    sobel_size = 3
    cv2.createTrackbar("sobel size", windowName2, sobel_size, 7, nothing)

    image = cv2.imread(sys.argv[1])

    while(keep_processing):

        lower_threshold = cv2.getTrackbarPos("lower", windowName2);
        upper_threshold = cv2.getTrackbarPos("upper", windowName2);
        smoothing_neighbourhood = cv2.getTrackbarPos("smoothing", windowName2);
        sobel_size = cv2.getTrackbarPos("sobel size", windowName2);

        smoothing_neighbourhood = max(3, smoothing_neighbourhood)
        if not(smoothing_neighbourhood % 2):
            smoothing_neighbourhood = smoothing_neighbourhood + 1

        sobel_size = max(3, sobel_size)
        if not(sobel_size % 2):
            sobel_size = sobel_size + 1

        gray_frame = illuminationHSV(image)

        # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())

        smoothed = cv2.GaussianBlur(gray_frame, (smoothing_neighbourhood, smoothing_neighbourhood), 0)

        # perform canny edge detection

        edges = cv2.Canny(smoothed, lower_threshold, upper_threshold, apertureSize=sobel_size)

        cv2.imshow(windowName, image)
        cv2.moveWindow(windowName, 0, 144)

        cv2.imshow(windowName2, edges)
        cv2.moveWindow(windowName2, 640, 0)

        key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False;

    cv2.destroyAllWindows()

else:
    print("No image file specified.")
