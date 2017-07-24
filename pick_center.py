#!/usr/bin/env python2

# usage ( standalone ):   python pick_center.py --image input.jpg
# usage ( imported ): pick_center ( input.jpg )

import os
import logging
import argparse
import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

###
# SETUP VARIABLES
# Camera screen length is 350 mm and breath is 240mm while focus is taken from a height of 300mm
area_w = 350 # mm
area_h = 240 # mm

# minimum/maximum fraction of total area required
# Contour/shape to get considered
min_area = 0.005
max_area = 0.1

# to control the size of text and drawing
max_pix = 2000  # max pixel size for a dimension

###

def pick_center( path_image ):
    #read the image 
    image = cv2.imread( path_image )
    
    # find width and height of the image
    h = image.shape[0]
    w = image.shape[1]
    
    # factor to control size of text and drawings
    f_s = float(w)/max_pix
    # area of the image in square pixels
    im_area = h*w
    
    # origin of the image
    orig_x = int(w/2)
    orig_y = int(h/2)
    print ('origin: ', orig_x, orig_y)
 
    # setup multiplication factor

    mult_fact_w = float(area_w)/w  # mm/pixel
    mult_fact_h = float(area_h)/h # mm/pixel

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding helps in removing noise
    _,thresh = cv2.threshold(gray, 100,250,cv2.THRESH_BINARY_INV)   # experiment here

    # find opening of the gray/thresholded image
    # Opening is basically is the image with sharpened edges
    # So, we get much more clear shapes than original one
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Write back opening image in the drive
    cv2.imwrite('opened.jpg', opened)
    
    # Pre-processing for contour finding
    # Perform gaussian blurring and
    # then run Canny edge detector
    gray = cv2.GaussianBlur(opened, (5,5), 0)
    edged = cv2.Canny(gray, 10, 100)
    edged = cv2.GaussianBlur(edged, (5,5), 0)
    #
    # finally find contours
    im2, cnts, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours based on area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    ###
    # FIND CENTER OF CLOSEST HOLE
    ###
    
    # initialize closest center ( edge of the image )
    closest_center = math.sqrt( orig_x**2 + orig_y**2)
    closest_cont = None
    for c in cnts:
        # Area of chosen contour
        cont_area = cv2.contourArea(c)
        
        # Pick only if the area of contour is acceptable 
        if (cont_area > min_area*im_area ) and (cont_area < max_area*im_area):
            cv2.drawContours(image, [c], -1, (0, 0, 255), max(1, int(5*f_s)))
            
            # find center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # distance of center of contour from origin
            center_dist = math.sqrt( (orig_x - cX)**2 + (orig_y - cY)**2)
            
            if center_dist < closest_center:
                closest_center = center_dist
                closest_cont = c
                # draw the center of the contour
                cv2.circle(image, (cX, cY), int(10*f_s), (0, 0, 255), -1)
                # 2 is text size, 7 is size of the line writing text
                text_size = max( 1, int(2*f_s))
                text_width = max( 1, int(7*f_s)) 
                cv2.putText(image, "Hole Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, text_size , (255, 0, 255),text_width )
                cent_X = cX
                cent_Y = cY
                
    # highlight the origin too
    cv2.circle(image, (int(w/2), int(h/2)), max( 1, int(20*f_s)), (0, 255, 255), -1)
    cv2.putText(image, "Origin", (int(w/2) -50, int(h/2)-50), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 255), text_width)
    
    # print the chosen contor
    cv2.drawContours(image, [closest_cont], -1, (0, 255, 255), max( 1, int(10*f_s)))
    cv2.imwrite('contours.jpg', image)

    ## return the relative Gcode
    rel_x = mult_fact_w*( cent_X - orig_x)
    # in CV downward in considered positive 
    # multiply by -1 is to negate that
    rel_y = mult_fact_h*(-1)*( cent_Y - orig_y)
    return rel_x, rel_y


if __name__ == '__main__':
    # parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
    args = vars(ap.parse_args())
    
    # Call the pick_center function to get closest hole's center
    mov_x, mov_y = pick_center( args["image"] )
    print ('Move right from origin(mm): %.4f ' %(mov_x))
    print ('Move up from origin(mm): %.4f ' %(mov_y))

