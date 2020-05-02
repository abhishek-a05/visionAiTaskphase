import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# press 'q' *to exit from the video
cap = cv.VideoCapture(0)


def nothing(x):
    pass


cv.namedWindow('Trackbar')
cv.createTrackbar('contour area', 'Trackbar', 0, 2000, nothing)
while True:
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # set trackbar for varying area
    contour_area = cv.getTrackbarPos('contour area', 'Trackbar')

    # calculate the difference in subsequent frames
    diff = cv.absdiff(frame1, frame2)

    # convert to grayscale for processing
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    # blur to eliminate the noise with gaussian kernel size 5x5
    blur = cv.GaussianBlur(diff_gray, ksize=(5, 5), sigmaX=0)

    # apply threshold to blur to eliminate unnecessary colors
    ret, thresh = cv.threshold(blur, thresh=10, maxval=255, type=cv.THRESH_BINARY)

    # define kernel to
    # erode and dilate the black pixels within the frame
    kernel = np.ones((5, 5), np.uint8)
    thresh_open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    # find contours for edge points in the object using chain approximation method
    contour, hierarchy = cv.findContours(thresh_open, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # to plot a bounding rectangle
    for i in range(len(contour)):
        # obtain the coordinates for the bounding rectangle ie x,y,width,height
        (x, y, w, h) = cv.boundingRect(contour[i])
        print(cv.contourArea(contour[i]))
        if cv.contourArea(contour[i]) < contour_area:
            continue
        else:
            cv.rectangle(frame1, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 255), thickness=2)
            cv.putText(frame1, text='motion detected', org=(10, 20),
                       fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

    # display frame to view
    cv.imshow('frame', frame1)

    # assign new frame to previous frame
    frame1 = frame2

    # re-define new frame
    ret, frame2 = cap.read()

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
