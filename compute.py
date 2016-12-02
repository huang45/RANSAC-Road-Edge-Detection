'''
    File name: compute.py
    Author: jrnf24@durham.ac.uk
    Date created: 6/11/2016
    Date last modified: 02/12/2016
    Python Version: 3.5.2
    Info: Performs RANSAC detection upon input images in order to identify
        the presence of road edges/markings.
    Instructions: The program cycles through the 'directory_to_cycle' directory,
        outputs to the terminal and displays image outputs. Change the
        variable to use a different directory.
    Acknowledgements: 'Canny Edge Example - Lecture 2' Toby Breckon, toby.breckon@durham.ac.uk
        'cycleimages.py' Toby Breckon, toby.breckon@durham.ac.uk
        OpenCV library, http://opencv.org
'''

import cv2
import sys
import numpy as np
import random
import os

directory_to_cycle = "/Users/Andrew/Documents/Course/SSA/CompVis/Assignment/data"



def preProcess(image):
    ''' Performs pre-processing on an input image so it
        is ready to be used by the RANSAC algorithm '''
    # Perform illumination invariant transform
    filtered = illuminationHSV(image)
    # Smooth image
    smoothed = cv2.GaussianBlur(filtered, (3, 3), 0)
    # Perform canny edge detection
    edges = cv2.Canny(smoothed, 25, 120, apertureSize=3)
    # If we have a messy edge image, apply more strict filtering
    if len(np.nonzero(edges)[0]) > 30000:
        edges = cv2.Canny(smoothed, 75, 180, apertureSize=3)
    # Ignore edges in parts of image by obscuring them
    rows,cols = edges.shape
    cv2.rectangle(edges,(0,0),(cols,int(5*rows/8)),(0,0,0),-1,8)
    cv2.rectangle(edges,(0,0),(int(cols/8),rows),(0,0,0),-1,8)
    cv2.line(edges, (0,380), (300,240), (0,0,0), 135)
    return edges



def illuminationHSV(image):
    ''' Takes an BGR image and returns the
        saturation component of the equivalent
        image in HSV space '''
    rows,cols,channels = image.shape
    # Convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split channels
    h,s,v = cv2.split(image_hsv)
    # If this is a bright image then just convert it into grayscale,
    # otherwise use the S component of the HSV equivalent image
    if np.average(v) < 125:
        return s
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def ransac(image, edges):
    ''' Performs the RANSAC algorithm upon an input
        image and its corresponding pre-processed
        image, outputting an image with lines drawn
        on it and printing the amount of lines to stdout '''
    # Initialise variables
    rows,cols,channels = image.shape
    white = np.nonzero(edges)
    length = len(white[0])
    densestLine = -1
    bestInliers = 0
    lineCounter = 0
    iterations = 10000
    # While we have strong lines (probably road edges), enough edges and not
    # more than 4 lines (roads in UK don't have more 4)
    while bestInliers > (densestLine / 2) and lineCounter < 5 and length > 1000:
        # Build a bank of the locations of edge pixels
        white = np.nonzero(edges)
        length = len(white[0])
        # Initialise variables
        bestInliers = 0
        bestLine = []
        # Find lines, updating best if beaten
        for i in range (iterations + 1):
            # Select two random points
            rand1 = random.randrange(length)
            rand2 = random.randrange(length)
            x1,y1 = white[1][rand1],white[0][rand1]
            x2,y2 = white[1][rand2],white[0][rand2]
            # Heuristic to eliminate vertical and horizontal lines
            # keep selecting points until heuristics aren't violated
            cycles = 0
            while abs(x1 - x2) < 50 or abs(y1 - y2) < 50 or abs(x1 - x2) > 200:
                rand1 = random.randrange(length)
                rand2 = random.randrange(length)
                x1,y1 = white[1][rand1],white[0][rand1]
                x2,y2 = white[1][rand2],white[0][rand2]
                cycles += 1
                if cycles > 100000:
                    break
            # Create blank image
            lineImage = np.zeros((rows,cols), np.uint8)
            # Draw line between two points on blank image
            cv2.line(lineImage, (x1,y1), (x2,y2), (255,255,255), 1)
            # Logically AND the line image and processed image
            intersect = cv2.bitwise_and(edges, lineImage)
            # Calculate the number of inliers to the line
            inliers = cv2.countNonZero(intersect)
            # Modify best if necessary
            if inliers >= bestInliers:
                bestInliers = inliers
                bestLine = [(x1,y1), (x2,y2)]
            if cycles > 100000:
                break
        # Update global best to help know when to stop
        if bestInliers > densestLine:
            densestLine = bestInliers
        # Only draw line if it's a strong line (potential road edge)
        if bestInliers > (densestLine / 2):
            # Draw the line on the image
            cv2.line(image, bestLine[0], bestLine[1], (0,0,255), 1)
            # Prevent choosing this same line again my removing edges
            cv2.line(edges, bestLine[0], bestLine[1], (0,0,0), 75)
            lineCounter += 1
        if cycles > 100000:
            break
    return [image, lineCounter]


for filename in os.listdir(directory_to_cycle):
    # if it is a PNG file
    if '.png' in filename:
        # read it and display in a window
        image = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
        # Perform pre-processing to ready input for RANSAC
        edges = preProcess(image)
        # Perform RANSAC algorithm
        output,lineCounter = ransac(image, edges)
        print ("%s : detected %d edges/lines" % (filename, lineCounter))
        cv2.imshow('RANSAC Detection', output)

        key = cv2.waitKey(200) # wait 200ms
        if (key == ord('x')):
            break

# close all windows
cv2.destroyAllWindows()
