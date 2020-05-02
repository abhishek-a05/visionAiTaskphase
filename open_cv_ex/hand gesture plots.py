import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

up = cv.imread('up.JPG', cv.IMREAD_GRAYSCALE)
down = cv.imread('down.JPG', cv.IMREAD_GRAYSCALE)
left = cv.imread('left.JPG', cv.IMREAD_GRAYSCALE)
right = cv.imread('right.JPG', cv.IMREAD_GRAYSCALE)
up_color = cv.imread('up_color.JPG', cv.IMREAD_COLOR)
down_color = cv.imread('down_color.JPG', cv.IMREAD_COLOR)
left_color = cv.imread('left_color.JPG', cv.IMREAD_COLOR)
right_color = cv.imread('right_color.JPG', cv.IMREAD_COLOR)

titles = ['up', 'down', 'left', 'right', 'up color', 'down color', 'left color', 'right color']

plot = [up, down, left, right, up_color, down_color, left_color, right_color]


for i in range(8):
        plt.subplot(2, 4, i + 1), plt.imshow(plot[i], 'gray')
        plt.title(titles[i])


plt.savefig('hand gesture legend.png',dpi=300)
plt.show()
