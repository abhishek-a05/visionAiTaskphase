import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

# PALM open - UP
# PALM close- DOWN
# PALM close and Thumb - LEFT
# PALM close and little finger- RIGHTq


cap = cv.VideoCapture(0)


def nothing(x):
    pass


# function to find distance between 2 points
def distance(coordinate1, coordinate2):
    x1 = coordinate1[0]
    y1 = coordinate1[1]
    x2 = coordinate2[0]
    y2 = coordinate2[1]

    dist = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

    return dist


# cv.namedWindow('Trackbar')
# cv.createTrackbar('lower hue', 'Trackbar', 0, 179, nothing)
# cv.createTrackbar('lower saturation', 'Trackbar', 0, 255, nothing)
# cv.createTrackbar('lower value', 'Trackbar', 0, 255, nothing)
# cv.createTrackbar('upper hue', 'Trackbar', 179, 179, nothing)
# cv.createTrackbar('upper saturation', 'Trackbar', 255, 255, nothing)
# cv.createTrackbar('upper value', 'Trackbar', 255, 255, nothing)
# cv.createTrackbar('threshold','Trackbar',0,255, nothing)


while True:
    _, frame = cap.read()
    cv.rectangle(frame, (10, 80), (250, 350), (255, 0, 255), 1)
    # define ROI as per the rectangle coordinates
    roi = frame[80:350, 10:250]

    roi_blur = cv.GaussianBlur(roi, (5, 5), 0)

    roi_hsv = cv.cvtColor(roi_blur, cv.COLOR_BGR2HSV)

    # lower_hue = cv.getTrackbarPos('lower hue', 'Trackbar')
    # lower_sat = cv.getTrackbarPos('lower saturation', 'Trackbar')
    # lower_val = cv.getTrackbarPos('lower value', 'Trackbar')
    # upper_hue = cv.getTrackbarPos('upper hue', 'Trackbar')
    # upper_sat = cv.getTrackbarPos('upper saturation', 'Trackbar')
    # upper_val = cv.getTrackbarPos('upper value', 'Trackbar')
    # thresh=cv.getTrackbarPos('threshold','Trackbar')

    # create upper and lower bounds for peach color
    lower_peach = np.array([0, 60, 0])
    upper_peach = np.array([179, 255, 255])

    # create a mask to detect color within upper and lower bound ranges
    mask = cv.inRange(roi_hsv, lower_peach, upper_peach)

    # filter the mask using dilation ,erosion and blurring
    kernel = np.ones((1, 1), np.uint8)
    mask_dilate = cv.dilate(mask, kernel, iterations=1)
    mask_erode = cv.erode(mask_dilate, kernel, iterations=1)

    mask_filter = cv.GaussianBlur(mask_erode, (3, 3), 0)
    ret, thresh = cv.threshold(mask_filter, 10, 255, 0)
    median_thresh = cv.medianBlur(thresh, 5)

    # define contours
    contours, hierarchy = cv.findContours(median_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # finding maximum area contour for bounding rectangle
    contour = max(contours, key=lambda x: cv.contourArea(x))
    epsilon = 0.00001 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # find coordinates of bounding rectangle
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # define centroid of white region
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centroid = (cx, cy)
    print('centroid:', centroid)
    print('\n')

    # draw the centroid on the ROI
    cv.circle(roi, (cx, cy), 10, (0, 0, 255), -1)

    cv.drawContours(roi, approx, -1, (255, 0, 255), 2)
    area = cv.contourArea(approx)

    # define right most and left most coordinates of maximum area contour
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    cv.circle(roi, leftmost, 10, (0, 0, 255), -1)
    cv.circle(roi, rightmost, 10, (0, 0, 255), -1)
    cv.line(roi, leftmost, (cx, cy), (0, 255, 0), 1)
    cv.line(roi, rightmost, (cx, cy), (0, 255, 0), 1)

    print('area:', area)
    print('\n')
    print('leftmost:', leftmost)
    print('\n')
    print('rightmost:', rightmost)
    diff = abs(distance(leftmost, centroid) - distance(rightmost, centroid))
    print('difference:', diff)

    # code block to classify gesture on the basis of area of contour and distance from centroid
    if area > 18000:
        cv.putText(frame, 'up', (260, 90), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        print('up\n')
    elif area < 10000:
        cv.putText(frame, 'move hand forward', (260, 90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    elif 10000 < area < 12500:
        cv.putText(frame, 'down', (260, 90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        print('down\n')
    else:
        if 12500 < area < 16000:
            if distance(leftmost, centroid) > distance(rightmost, centroid):
                cv.putText(frame, 'right', (260, 90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                print('right\n')
            elif distance(leftmost, centroid) < distance(rightmost, centroid):
                cv.putText(frame, 'left', (260, 90), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                print('left\n')

    # cv.circle(roi,topmost,3,(0,0,255),thickness=2)
    # cv.circle(roi,bottommost,3,(0,255,255),thickness=2)

    # show thresh and frame
    cv.imshow('roi', median_thresh)
    cv.imshow('image', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
