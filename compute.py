import cv2
import sys
import numpy as np
import math
import random

def illuminationAlg(image, alpha):
    rows,cols,channels = image.shape
    newImage = np.zeros((rows,cols,1), np.float)
    image = image.astype('float') / 255
    for x in xrange (rows):
        for y in xrange (cols):
            if image[x][y][0] == 0:
                image[x][y][0] = 1
            if image[x][y][1] == 0:
                image[x][y][1] = 1
            if image[x][y][2] == 0:
                image[x][y][2] = 1
            newValue = 0.5 + math.log(image[x][y][2]) - (alpha * math.log(image[x][y][1])) - ((1 - alpha) * math.log(image[x][y][0]))
            newImage[x][y] = newValue
    newImage = (newImage * 255).astype('uint8')
    return newImage

def illuminationHSV(image):
    rows,cols,channels = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.split(image_hsv)
    return hsv[1]

def ransac(image, edges):
    # Build a bank of the locations of edge pixels
    candidates = []
    rows,cols = edges.shape
    for i in xrange (rows):
        for j in xrange ((cols/2), cols):
            if edges[i][j] == 255:
                candidates.append([i,j])
    # Sample two random points
    length = len(candidates)
    iterations = 1
    current = 0
    best = 0
    bestLine = []
    while current < iterations:
        rand1 = rand2 = random.randrange(length) # MAY NOT BE TOTALLY RANDOM -- CHECK
        while rand1 == rand2:
            rand2 = random.randrange(length)
        x1,y1 = candidates[rand1]
        x2,y2 = candidates[rand2]
        lineImage = np.zeros((rows,cols), np.uint8)
        #line = cv2.fitLine(np.array([(x1,y1), (x2,y2)]), cv2.DIST_L2, 0, 0.01, 0.01)
        cv2.line(lineImage, (x1,y1), (x2,y2), (255,0,0), 3)
        #for x in xrange (rows):
        #    for y in xrange (cols):
        #        expected = (int)(((y2 - y1) / (x2 - x1)*(x - x1)) + y1 - y)
        #        if expected <= 2 and expected >= -2:
        #            blank[x][y] = 255
        intersect = cv2.bitwise_and(edges, lineImage)
        inliers = 0
        for x in xrange (rows):
            for y in xrange (cols):
                if intersect[x][y] == 255:
                    inliers += 1
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 1)
        if inliers >= best:
            best = inliers
            bestLine = [(x1,y1), (x2,y2)]
        current += 1
    cv2.line(image, bestLine[0], bestLine[1], (0,0,255), 5)
    return image

def hough(image, edges):
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines in locals():
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image


windowName = "Edge Detection"
windowName2 = "RANSAC Detection"
keep_processing = True

if (len(sys.argv) == 2):

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)

    lower_threshold = 110
    upper_threshold = 170
    smoothing_neighbourhood = 3
    sobel_size = 3

    image = cv2.imread(sys.argv[1])

    smoothing_neighbourhood = max(3, smoothing_neighbourhood)
    if not(smoothing_neighbourhood % 2):
        smoothing_neighbourhood = smoothing_neighbourhood + 1

    sobel_size = max(3, sobel_size)
    if not(sobel_size % 2):
        sobel_size = sobel_size + 1

    gray_frame = illuminationAlg(image, 0.5)
    windowName3 = "Test Illumination"
    cv2.namedWindow(windowName3, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName3, gray_frame)
    cv2.moveWindow(windowName3, 0, 0)

    # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())

    smoothed = cv2.GaussianBlur(gray_frame, (smoothing_neighbourhood, smoothing_neighbourhood), 0)

    # perform canny edge detection

    edges = cv2.Canny(smoothed, lower_threshold, upper_threshold, apertureSize=sobel_size)

    #_, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(edges, contours, -1, (255,0,0), 3)

    #copyImage = image.copy()
    #hough = hough(copyImage, edges)

    cv2.imshow(windowName, edges)
    cv2.moveWindow(windowName, 0, 0)

    ransac = ransac(image, edges)

    cv2.imshow(windowName2, ransac)
    cv2.moveWindow(windowName2, 640, 0)

    cv2.waitKey()

else:
    print("No image file specified.")
