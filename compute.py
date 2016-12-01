import cv2
import sys
import numpy as np
import random

def preProcess(image, smoothing_neighbourhood, lower_threshold, upper_threshold, sobel_size):
    filtered = illuminationHSV(image)
    smoothed = cv2.GaussianBlur(filtered, (smoothing_neighbourhood, smoothing_neighbourhood), 0)
    edges = cv2.Canny(smoothed, lower_threshold, upper_threshold, apertureSize=sobel_size)
    # Ignore edges in parts of image
    rows,cols = edges.shape
    cv2.rectangle(edges,(0,0),(cols,int(5*rows/8)),(0,0,0),-1,8)
    cv2.rectangle(edges,(0,0),(int(cols/8),rows),(0,0,0),-1,8)
    cv2.line(edges, (0,380), (300,240), (0,0,0), 135)
    cv2.imshow(windowName3, edges)
    return edges

# FORMULA IS ii_image = 0.5 + log(image(:,:,2)) - alpha*log(image(:,:,3)) - (1-alpha)*log(image(:,:,1))
# for RGB i.e. 0.5 + log(green) - alpha*log(blue) - (1 - alpha)*log(red)
def illuminationAlg(image, alpha):
    image = image.astype('float') / 255
    blue, green, red = cv2.split(image)
    green[green == 0] = 0.0039
    a = cv2.log(green)
    blue[blue == 0] = 0.0039
    b = cv2.multiply(cv2.log(blue), alpha)
    red[red == 0] = 0.0039
    c = cv2.multiply(cv2.log(red), (1 - alpha))
    newImage = cv2.subtract(cv2.subtract(cv2.add(0.5, a), b), c)
    #newImage[newImage == -np.inf] = 0
    #newImage[newImage == np.inf] = 1
    cv2.normalize(newImage, newImage, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    newImage = (newImage * 255).astype('uint8')
    cv2.normalize(newImage, newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return newImage

def illuminationHSV(image):
    rows,cols,channels = image.shape
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(image_hsv)
    #cv2.normalize(h, h, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    #cv2.normalize(s, s, alpha=0, beta=127, norm_type=cv2.NORM_MINMAX)
    #newImage = cv2.add(h,s)
    newImage = s
    #cv2.normalize(newImage, newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return newImage

def ransac(image, edges):
    rows,cols,channels = image.shape
    densestLine = -1
    bestInliers = 0
    lineCounter = 0
    while bestInliers > (densestLine / 2):
        # Build a bank of the locations of edge pixels
        white = np.nonzero(edges)
        length = len(white[0])
        iterations = 10000
        bestInliers = 0
        bestLine = []
        for i in range (iterations):
            rand1 = random.randrange(length)
            rand2 = random.randrange(length)
            x1,y1 = white[1][rand1],white[0][rand1]
            x2,y2 = white[1][rand2],white[0][rand2]
            # Heuristic to eliminate vertical and horizontal lines - may not work on some road edges e.g. 350
            while abs(x1 - x2) < 50 or abs(y1 - y2) < 25:
                rand1 = random.randrange(length)
                rand2 = random.randrange(length)
                x1,y1 = white[1][rand1],white[0][rand1]
                x2,y2 = white[1][rand2],white[0][rand2]
            lineImage = np.zeros((rows,cols), np.uint8)
            cv2.line(lineImage, (x1,y1), (x2,y2), (255,255,255), 1)
            intersect = cv2.bitwise_and(edges, lineImage)
            inliers = cv2.countNonZero(intersect)
            if inliers >= bestInliers:
                bestInliers = inliers
                bestLine = [(x1,y1), (x2,y2)]
        if bestInliers > densestLine:
            densestLine = bestInliers
        if bestInliers > (densestLine / 2):
            bestLine = extendLine(bestLine, bestInliers, edges)
            cv2.line(image, bestLine[0], bestLine[1], (0,0,255), 1)
            cv2.line(edges, bestLine[0], bestLine[1], (0,0,0), 75)
            lineCounter += 1
    print ("filename : detected %d edges/lines" % lineCounter)
    cv2.line(image, (50, rows - 50), (50, rows - 75), (0,0,0), 5)
    return image

def extendLine(points, inliers, edges):
    rows,cols = edges.shape
    x1,y1 = points[0]
    x2,y2 = points[1]
    if y1 > y2:
        firstX, firstY = x1, y1
        secondX, secondY = x2, y2
    else:
        firstX, firstY = x2, y2
        secondX, secondY = x1, y1
    gradient = (y2 - y1) / (x2 - x1)
    previous = inliers
    change = 100
    while change > 15:
        if firstX < cols - 5 and firstY < rows - (5*gradient):
            lineImage = np.zeros((rows,cols), np.uint8)
            cv2.line(lineImage, (firstX,firstY), (secondX,secondY), (255,255,255), 2)
            intersect = cv2.bitwise_and(edges, lineImage)
            newInliers = cv2.countNonZero(intersect)
            change = newInliers - previous
            firstX += 5
            firstY += int(gradient * 5)
            previous = newInliers
        else:
            break
    previous = inliers
    change = 100
    while change > 15:
        if secondX > 5 and secondY > (5*gradient):
            lineImage = np.zeros((rows,cols), np.uint8)
            cv2.line(lineImage, (firstX,firstY), (secondX,secondY), (255,255,255), 2)
            intersect = cv2.bitwise_and(edges, lineImage)
            newInliers = cv2.countNonZero(intersect)
            change = newInliers - previous
            secondX -= 5
            secondY -= int(gradient * 5)
            previous = newInliers
        else:
            break
    return [(firstX,firstY), (secondX,secondY)]

def hough(image, edges):
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
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


windowName = "Processed Image"
windowName2 = "RANSAC Detection"
windowName3 = "Test Illumination"
windowName4 = "Edges after lines"
windowName5 = "Hough Transform"

if (len(sys.argv) == 2):

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)

    lower_threshold = 25
    upper_threshold = 120
    smoothing_neighbourhood = 3
    sobel_size = 3

    image = cv2.imread(sys.argv[1])

    smoothing_neighbourhood = max(3, smoothing_neighbourhood)
    if not(smoothing_neighbourhood % 2):
        smoothing_neighbourhood = smoothing_neighbourhood + 1

    sobel_size = max(3, sobel_size)
    if not(sobel_size % 2):
        sobel_size = sobel_size + 1

    edges = preProcess(image, smoothing_neighbourhood, lower_threshold, upper_threshold, sobel_size)

    #copyImage = image.copy()
    #hough = hough(copyImage, edges)

    cv2.imshow(windowName, edges)
    cv2.moveWindow(windowName, 0, 0)

    ransac = ransac(image, edges)

    cv2.imshow(windowName2, ransac)
    cv2.moveWindow(windowName2, 640, 0)
    '''
    hough = hough(image, edges)
    cv2.namedWindow(windowName5, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName5, hough)
    cv2.moveWindow(windowName5, 0, 0)
    '''
    cv2.waitKey()

else:
    print("No image file specified.")
