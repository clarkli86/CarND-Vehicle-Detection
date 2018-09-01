
# coding: utf-8

# ## 1. Camera Calibration
# OpenCV has provided a method to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images. The following code snippet uses the chessboard images (9x6) in repository to calibrate the camera. Example code opencv2.org has been modified to fit new chessboard size.

# In[1]:


import cv2
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_camera():
    """
    Calibrate camera using pre-captured images
    https://docs.opencv2.org/3.4.0/dc/dbb/tutorial_py_calibration.html
    """
    nx = 6
    ny = 9
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    """Undistort an image using camera matrix and distortion coefficients"""
    h,  w = img.shape[:2]
    # return undistorted image with minimum unwanted pixels. It's okay to remove some pixesl at image corners.
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undist

def test_calibration(mtx, dist, filename):
    """Util function to test camera calibration"""
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = undistort(img, mtx, dist)

# #### Test calibration on the selected images
#
# It is evident that camera distortion in `camera_cal/calibration1.jpg` and `camera_cal/calibration2.jpg` has been successfully corrected.
#
# We should be confident that the calibrated camera will help us reduce the errors in lane detection and radius curvature calculation.

# In[2]:


# Calibrate and test on multiple images
mtx, dist = calibrate_camera()

# ## 2. Pipeline for Static Images
# My pipeline for Static Images includes the following steps:
# 1. Undistort image.
# 1. Create Threshold (color/gradient) Binary from undistroted image.
# 1. Perform Perspective Transform on Threshold Binary to convert it to birds-eye View.
# 1. In the birds-eye View of Threshold Binary, find lane lines using moving windows. Also calculate the radius of curvature and vehicle position. Fill the area within lane lines with solid color.
# 1. Perform Perspective to convert the Threshold Binary into the original view.
# 1. Overlay the Threshold Binary to the original image.
# 1. Plot radius of curvature and vehicle position calculated in step 4 at the top-left corner of the original image.
#
# Next I am going to describe each step in detail. `test_images_undistorted/straight_lines1.jpg` is used to demonstrate effect of each step.
#
# <center>Original Image</center>
# <img src="test_images/straight_lines1.jpg" width="300"/>

# ### 2.1. Undistort Image
# Undistort image using the camera matrix and distortation coefficients.

# In[3]:

# ### 2.2. Threshold Binary based on Color/Gradient
#
# Two different approaches have been considered for threshold binary -
# 1. Combination of x/y gradients and gradient magnitude
# 1. Combination of HLS color selection and x gradient
#
# Result in the block below suggets that `HLS + RGB + x gradient` picks up more features from lane lines.

# In[4]:


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def rgb_select(img, thresh=(0, 255)):
    r_channel = img[:,:,0]
    binary_output = np.zeros_like(r_channel)
    binary_output[(r_channel > thresh[0]) & (r_channel <= thresh[1])] = 1
    return binary_output

def combined_gradient_mag(img):
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

def combined_color_gradient(img):
    # Threshold x gradient
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=40, thresh_max=100)
    # Threshold color channel
    s_binary = hls_select(img, (90, 255))
    r_binary = rgb_select(img, (200, 255))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[((r_binary == 1) & (s_binary == 1)) | (gradx == 1)] = 1
    return combined_binary

# ### 2.3. Perform Perspective Transform to birds-eye View
#
# A polygon, which in birds-view should be a perfect rectangle, is constructed by
# connecting the straight lane lines. The vertices of this polygon in camera view and birds-eye view are used to
# find the transform matrix.

# In[5]:


def perspective_transform_to_from_birds_eye(img):
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100 # offset for dst points
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    src = np.float32([(590, 460), (709, 460), (1093, 719), (230, 719)])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([[300, 0], [980, 0], [980, img_size[1]-1], [300, img_size[1]-1]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_ = cv2.getPerspectiveTransform(dst, src)
    return M, M_

def warp(img, M):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped

# Use straight lines to get perspective transform matrix
undist_straight_lines = mpimg.imread("straight_lines1_marked.jpg")
warp_to_birds_eye_M, warp_from_birds_eye_M = perspective_transform_to_from_birds_eye(undist_straight_lines)

# ### 2.4. Find lane-line pixels and fit their positions with a polynomial
#
# Searching for the lane-line pixels is done from bottm to top.
#
# Firstly a histogram with two peaks is calculated to locate the bottom of both lanes.
#
# Secondly a window of 200x80 is moved up along the lane lines to locate the next segment. In every substep, mean of the horizontal positions is calculated on pixels within the current window. This mean position will be used as horizontal center of the next window.
#
# Finally pixels in these windows are used to fit two polynomilas - one for the left lane line and one for the right lane line.

# In[6]:


def find_lane_pixels(binary_warped, debug=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if debug:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, debug=False):
    """Fit lane pixels in image to polynomials"""
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, debug=debug)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if debug:
        # Plots the left and right polynomials on the lane lines
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img

def fill_polynomial(binary_warped, left_fit, right_fit, debug=False):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    polynomialgon = binary_warped.copy()

    # Create an image to draw the lines on
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(polynomialgon, np.int_([pts]), (0,255, 0))

    return polynomialgon

def debug_polynomial(left_fit, right_fit, leftx, lefty, rightx, righty, out_img):
    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

# After fitting polynominals in pixels,
#
# 1. We try to calculte the radious of curvature in metre. **
# For this project, we can assume that if we're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. (https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/096009a1-3d76-4290-92f3-055961019d5e/concepts/1a352727-390e-469d-87ea-c91cd78869d6)
#
# Thus we define the following convertion factors
# ```python
# # Number of lane pixels in y dimenstion
# num_pix_y = 720
# # Number of lane pixels in x dimension (roughly 1280 - 600)
# num_pix_x = 680
# # Define conversions in x and y from pixels space to meters
# ym_per_pix = 30/num_pix_y # meters per pixel in y dimension
# xm_per_pix = 3.7/num_pix_x # meters per pixel in x dimension
# ```
#
# 2. We also try to calculate the vehicle position relative to the lane center. This is done by finding the lane center in the original image then calculate its offset to image center. Here we assume the camera is mounted at the center of the car.

# In[7]:


def measure_curvature(img, leftx, lefty, rightx, righty, xm_per_pix=1.0, ym_per_pix=1.0):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Fit a second order polynomial to each using `np.polyfit`
    try:
        left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        y_eval = img.shape[0] - 1
        left_curverad = ((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**1.5) / (2*left_fit_m[0])
        right_curverad = ((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**1.5) / (2*right_fit_m[0])
    except:
        left_fit_m = [0, 0, 0]
        right_fit_m = [0, 0, 0]
        left_curverad = 3000
        right_curverad = 3000

    return left_curverad, right_curverad, left_fit_m, right_fit_m

def measure_distance(img, left_fit, right_fit, xm_per_pix=1.0, ym_per_pix=1.0):
    '''
    Calculates the distance of vehicle to the center of original image. left -> negative, right->positive
    '''
    y = img.shape[0] * ym_per_pix
    # Find two x coordiantes at the bottom of image in transformed image
    left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    center_x = (left_x + right_x) / 2

    center_x_image = img.shape[1] / 2 * xm_per_pix
    return center_x_image - center_x

# Number of lane pixels in y dimenstion
num_pix_y = 720
# Number of lane pixels in x dimension (roughly 1280 - 600)
num_pix_x = 680
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/num_pix_y # meters per pixel in y dimension
xm_per_pix = 3.7/num_pix_x # meters per pixel in x dimension

# ### 2.5. Perform Perspective to convert the Filled Lane Area into the original view.
#
# Using `perspective_transform_from_birds_eye()` in step 3, but this time we swap `src` and `dst` of the transform matrix. The output of `cv2.getPerspectiveTransform()` will transform an image from birds-eye view to the original camera view.

# In[8]:

# ### 2.6. Overlay the Filled Lane Area/Radius of Curvature/Vehicle Position to the original image
# In this step we use the `weighted_img()` function from `CarND-LaneLines-P1` (https://github.com/udacity/CarND-LaneLines-P1)

# In[9]:


def weighted_img(initial_img, img, α=1., β=0.8, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# ### 2.7. Plot radius of curvature and vehicle position calculated in step 2.4

# In[10]:


def plot_curvature(img, radius):
    '''
    Plot the curvature of polynomial functions in pixels.
    '''
    out_img = img.copy()
    cv2.putText(out_img, 'Radius of Curvature: {0:.2f} m'.format(radius), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return out_img

def plot_curvatures(img, left_radius, right_radius):
    '''
    Plot the curvature of polynomial functions in pixels.
    '''
    out_img = img.copy()
    cv2.putText(out_img, 'Radius of Left Curvature: {0:.2f} m'.format(left_radius), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(out_img, 'Radius of Right Curvature: {0:.2f} m'.format(right_radius), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return out_img

def plot_distance(img, distance):
    '''
    Plot vehicle position offset to the center of lane.
    '''
    out_img = img.copy()
    cv2.putText(out_img, 'Distance to center: {0:.2f} m'.format(distance), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return out_img

# ### Pipeline for static images
# With the components above, we can easily build up a pipeline to process static images. Let's apply them to all the test images.

# In[11]:


def save_test_image(img, filename):
    outdir = os.path.dirname(filename)
    try:
        os.mkdir(outdir)
    except:
        pass

    # cv2 has flipped bits - https://stackoverflow.com/questions/42406338/why-cv2-imwrite-changes-the-color-of-pics
    if len(img.shape) > 2:
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(filename, cv2.cvtColor(img * 255, cv2.COLOR_GRAY2BGR))

def pipeline(img, save_intermediate=False, fname=''):
    undist = undistort(img, mtx, dist)
    threshold_binary = combined_color_gradient(undist)
    birds_eye = warp(threshold_binary, warp_to_birds_eye_M)
    left_fit, right_fit, leftx, lefty, rightx, righty, out_img = fit_polynomial(birds_eye, debug=False)
    filled_poly = fill_polynomial(out_img, left_fit, right_fit, debug=False)
    restored = warp(filled_poly, warp_from_birds_eye_M)
    overlayed = weighted_img(undist, restored)

    # Poly fit with real world metrics
    left_radius_m, right_radius_m, left_fit_m, right_fit_m = measure_curvature(filled_poly, leftx, lefty, rightx, righty, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)
    radius_m = max(np.absolute(left_radius_m), np.absolute(right_radius_m))
    distance_m = measure_distance(filled_poly, left_fit_m, right_fit_m, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)
    weighted_curvature = plot_curvature(overlayed, radius_m)
    weighted_distance = plot_distance(weighted_curvature, distance_m)

    if save_intermediate:
        filename = os.path.basename(fname)
        save_test_image(undist, 'output_images/undistort/' + filename)
        save_test_image(threshold_binary, 'output_images/threshold/' + filename)
        save_test_image(birds_eye, 'output_images/birds_eye/' + filename)
        save_test_image(filled_poly, 'output_images/filled_poly/' + filename)
        save_test_image(restored, 'output_images/restored/' + filename)
        save_test_image(overlayed, 'output_images/overlayed/' + filename)
        save_test_image(weighted_distance, 'output_images/final/' + filename)

    return weighted_distance

# ### Pipeline for video
# The pipeline described above works pretty well for static images. To process video with many frames in real time,
# however, some components need to work more efficiently:
# 1. When looking for lane line pixels in **Step 4**, there is no need to run `histogram()` from the bottom. A better approach is to reuse the polynominal from last frame. In the new frame, we firstly find all the pixels within a horizontal margin around the last poly line. Then these pixels will be used to fit the new polynominal. The old approach will still be applied if a new polynominal couldn't be fit.
# 1. A sanity check is performed upon left and right radius of curvature. If they are going different directions or the difference is too big, `histogram()` will be applied to restart the moving windows from bottom of image.
# 1. An average filter is applied to the last five polynominal fits. The filled region becomes less woblly with the averaged coefficients.

# In[12]:


class SearchAroundPoly(object):
    def __init__(self):
        # Initialise left_fit and right_fit
        self.left_fit = [0, 0, 0]
        self.right_fit = [0, 0, 0]

    def search_around_poly(self, binary_warped, debug=False):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy +
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) +
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy +
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) +
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        if debug:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Create an image to draw on and an image to show the selection window
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                      ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                      ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')

            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            ## End visualization steps ##

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped, force_restart=False, debug=False):
        """Fit lane pixels in image to polynomials"""
        if (all(x == 0 for x in self.left_fit) and all(x == 0 for x in self.right_fit)) or force_restart:
            # Find our lane pixels first
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, debug=debug)
        else:
            leftx, lefty, rightx, righty, out_img = self.search_around_poly(binary_warped, debug=debug)

        left_radius_m, right_radius_m, left_fit_m, right_fit_m = measure_curvature(out_img, leftx, lefty, rightx, righty, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)

        # Reset when left and right don't agree
        if (np.absolute(left_radius_m - right_radius_m) > 1000 and np.absolute(left_radius_m) < 5000 and np.absolute(right_radius_m) < 5000) or            (np.absolute(left_radius_m) < 150) or            (np.absolute(right_radius_m) < 150) or            (left_radius_m < 0 and right_radius_m > 0) or            (left_radius_m > 0 and right_radius_m < 0):
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, debug=debug)
        try:
            # Fit a second order polynomial to each using `np.polyfit`
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
        except:
            self.left_fit = [0, 0, 0]
            self.right_fit = [0, 0, 0]

        return self.left_fit, self.right_fit, leftx, lefty, rightx, righty, left_radius_m, right_radius_m, left_fit_m, right_fit_m, out_img


# **ROI based on polynominals of the last frame**
#
# Pixels in this region will be used to fit the new polynominal.

# In[13]:

# In[14]:

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, h=720):
        # Max number of filters
        self.num_samples = 5
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #scale of y
        self.h = h

    def process(self, new_fit):
        self.current_fit = new_fit

        ploty = np.linspace(0, self.h-1, self.h)
        fitx = new_fit[0]*ploty**2 + new_fit[1]*ploty + new_fit[2]

        if len(self.recent_xfitted) >= self.num_samples:
            self.recent_xfitted = self.recent_xfitted[1:]
        self.recent_xfitted.append(fitx)

        if len(self.recent_xfitted) > 1:
            self.bestx = np.average(self.recent_xfitted, axis=0)
        else:
            self.bestx = fitx

        self.best_fit = np.polyfit(ploty, self.bestx, 2)

        return self.best_fit

class RadiusFilter():
    """Smooth radius"""
    def __init__(self):
        self.radius_m = float('nan')

    def process(self, radius_m):
        if math.isnan(self.radius_m):
            self.radius_m = radius_m
        else:
            self.radius_m = self.radius_m * 0.5 + radius_m * 0.5

        if self.radius_m > 3000:
            self.radius_m = 3000
        return self.radius_m


# In[15]:


search = SearchAroundPoly()
left_line = Line()
right_line = Line()
radius_filter = RadiusFilter()

def pipeline_video(img):
    undist = undistort(img, mtx, dist)
    threshold_binary = combined_color_gradient(undist)
    birds_eye = warp(threshold_binary, warp_to_birds_eye_M)
    left_fit, right_fit, leftx, lefty, rightx, righty,     left_radius_m, right_radius_m, left_fit_m, right_fit_m, out_img = search.fit_polynomial(birds_eye, debug=False)
    left_fit = left_line.process(left_fit)
    right_fit = right_line.process(right_fit)

    filled_poly = fill_polynomial(out_img, left_fit, right_fit, debug=False)
    warped_from_birds_eye = warp(filled_poly, warp_from_birds_eye_M)
    weighted = weighted_img(undist, warped_from_birds_eye)

    radius_m = max(np.absolute(left_radius_m), np.absolute(right_radius_m))
    distance_m = measure_distance(filled_poly, left_fit_m, right_fit_m, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)

    radius_m = radius_filter.process(radius_m)
    weighted_curvature = plot_curvature(weighted, radius_m)
    weighted_distance = plot_distance(weighted_curvature, distance_m)

    #cv2.imwrite('test_videos_output/frame' + str(index) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('test_videos_output/threshold_binary' + str(index) + '.jpg', cv2.cvtColor(threshold_binary * 255, cv2.COLOR_GRAY2RGB))
    #cv2.imwrite('test_videos_output/birds_eye' + str(index) + '.jpg', cv2.cvtColor(birds_eye * 255, cv2.COLOR_GRAY2RGB))
    #cv2.imwrite('test_videos_output/filled_poly' + str(index) + '.jpg', cv2.cvtColor(filled_poly, cv2.COLOR_BGR2RGB))
    #cv2.imwrite('test_videos_output/output' + str(index) + '.jpg', cv2.cvtColor(weighted_distance, cv2.COLOR_BGR2RGB))

    return weighted_distance

