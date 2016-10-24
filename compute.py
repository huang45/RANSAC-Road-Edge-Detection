import cv2
import sys

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass

windowName = "Original Image"; # window name
windowName2 = "New Image"; # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (len(sys.argv) == 2):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL);

    # add some track bar controllers for settings

    lower_threshold = 25;
    upper_threshold = 120;
    smoothing_neighbourhood = 3;
    sobel_size = 3; # greater than 7 seems to crash

    frame = cv2.imread(sys.argv[1])

    smoothing_neighbourhood = max(3, smoothing_neighbourhood);
    if not(smoothing_neighbourhood % 2):
        smoothing_neighbourhood = smoothing_neighbourhood + 1;

    sobel_size = max(3, sobel_size);
    if not(sobel_size % 2):
        sobel_size = sobel_size + 1;

    # convert to grayscale

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

    # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())

    smoothed = cv2.GaussianBlur(gray_frame,(smoothing_neighbourhood,smoothing_neighbourhood),0);

    # perform canny edge detection

    canny = cv2.Canny(smoothed, lower_threshold, upper_threshold, apertureSize=sobel_size);

    # display image

    cv2.imshow(windowName,frame);
    cv2.imshow(windowName2,canny)

    cv2.waitKey()

else:
    print("No image file specified.");
