import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]      # bottom of the image #index 0 is for heights
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit    = [] # contains coordinates of lines of left side of lane
    right_fit   = [] # right side
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)#polynomial of degree 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0: # y is reversed in image
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #STEP 1 : RGB TO GRAYSCALE
    blur = cv2.GaussianBlur(gray, (5,5), 0) # 5X5 kernel inserted to smoother img
    canny = cv2.Canny(blur, 50, 150) # for gradient image----indicates sharp change in intensities
    return canny

def display_lines(image,lines): # lines is a 3D array (as it is an array of lines and each line is a 2d array 1x4)
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # x1, y1, x2, y2 = line.reshape(4)  converted 2d array of a line to 1D array.....this step only when we were showing hough lines , now we are showing averaged lines so no need of reshape now.....
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10) #255,0,0 is blue color 10 is the thickness
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[   #assuming polygons array with one polygon our required triangle
    (200, height),(1100 , height),
    (550, 250)]])
    mask = np.zeros_like(image) # a new img with full black screen
    cv2.fillPoly(mask, polygons, 255) #we will fill our mask with our polygon
    masked_image = cv2.bitwise_and(image, mask) # to print only desired region
    return masked_image

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image);
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # precision of 2px by 1 radian with threshhold of 100
    # 5th argument in above line is just palce holder array
    # 6th argument is length of line in pixels that we'll accept in our output
    # 7th argument is maximum gap between lines that we'll allow to connect and form a line
    # be careful in choosing size of bins
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()): # same code as image it works for each frame of video
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image);
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'): # stops when q is pressed to break out of loop
        break
cap.release()
cv2.destroyAllWindows()
