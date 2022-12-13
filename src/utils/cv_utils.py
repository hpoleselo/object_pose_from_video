import cv2
import numpy as np
from collections import deque

frame = 0

def mouse_callback(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        
        blue_channel = frame[y, x , 0]
        green_channel = frame[y, x , 1]
        red_channel = frame[y, x, 2]
        print(f"RGB: ({red_channel},{green_channel},{blue_channel}); Coordinates (x,y): ({x},{y}).\n")

def open_video_with_mouse_callback(video_file: str):
    global frame
    print("Opening video")

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")


    # Read until video is completed
    while (cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('frame', frame)
            #cv2.namedWindow('mouseRGB')
            cv2.setMouseCallback('frame', mouse_callback)
            
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    # Break the loop
        else:
            break
    
    # When everything done, release
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def frame_rgb_to_hsv(frame):
    # Convert BGR to HSV
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def mask_hsv_with_color(
                        hsv_frame,             
                        color_lower_limit_in_hsv,
                        color_upper_limit_in_hsv,
                        offline_testing=False):
    """
    From clicking in the image, lower and upper limits were extractedd
    and then checked in: https://www.w3schools.com/colors/colors_rgb.asp
    """
    if offline_testing:
        print(f"Applying Minimum H: {color_lower_limit_in_hsv[0]}, S: {color_lower_limit_in_hsv[1]}, V:{color_lower_limit_in_hsv[2]}")
        print(f"Applying Maximum H: {color_upper_limit_in_hsv[0]}, S: {color_upper_limit_in_hsv[1]}, V:{color_upper_limit_in_hsv[2]}")
    lower_limit = np.array(color_lower_limit_in_hsv)
    upper_limit = np.array(color_upper_limit_in_hsv)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    # Bitwise-AND mask and original image
    masked_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    return masked_frame

def bgr2hsv_numerical(input_color_array_in_rgb: list):
    if len(input_color_array_in_rgb) != 3:
        print("Make sure to input an array of 3 digits, e.g: [0, 255, 0] to represent the RGB channels.")
    color_in_hsv = cv2.cvtColor(np.uint8([[input_color_array_in_rgb]]), cv2.COLOR_BGR2HSV)
    print(color_in_hsv)
    return color_in_hsv

def mask_image_test(img_file_path, test_with_static_image):
    # Prints out values to be inputted for mask_hsv_with_color()
    #get_hsv_values_for_red_box()

    # Reads in BGR
    img = cv2.imread(img_file_path)

    hsv = frame_rgb_to_hsv(img)

    # Masking by a red color
    # For the simple red img test
    #hsv_masked = mask_hsv_with_color(hsv, [0, 205, 215], [0, 226, 223])

    # Didnt workout
    #hsv_masked = mask_hsv_with_color(hsv, [2, 162, 60], [0, 227, 200])
    
    hsv_lower, hsv_upper = define_hsv_limits_from_bgr([24, 22, 84], is_static_image=test_with_static_image)
    hsv_masked = mask_hsv_with_color(hsv, hsv_lower, hsv_upper)

    cv2.imshow('original', img)
    cv2.imshow('masked', hsv_masked)
    cv2.waitKey(0)

def get_image_absolute_path(file_name: str):
    import os
    """
    path = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(path, os.pardir))
    test_data_path = os.path.join(parent_directory, 'test_data')"""
    # Better approach using Path
    from pathlib import Path
    # Gets current directory two levels up
    root_directory = os.path.join(Path(__file__).parents[2], 'test_data')
    return os.path.join(root_directory, file_name)

def define_hsv_limits_from_bgr(bgr_reference_color, is_static_image=False, is_lane_detection=False):
    bgr_reference_color = np.uint8([[bgr_reference_color]])
    hsv_reference_color = cv2.cvtColor(bgr_reference_color, cv2.COLOR_BGR2HSV)

    if is_static_image:
        if is_lane_detection:
            # * Using an offset for yellow
            print(f"Using HSV reference color as: {hsv_reference_color}")
            hsv_lower_offset = [hsv_reference_color[0][0][0] - 10, hsv_reference_color[0][0][1] - 70, hsv_reference_color[0][0][2] - 70]
            print(f"Lower offset: {hsv_lower_offset}")
            hsv_upper_offset = [hsv_reference_color[0][0][0] + 10, hsv_reference_color[0][0][1] + 70, hsv_reference_color[0][0][2] + 70]
            print(f"Higher offset: {hsv_upper_offset}")
        else:
            print(f"Using HSV reference color as: {hsv_reference_color}")
            hsv_lower_offset = [hsv_reference_color[0][0][0] - 10, hsv_reference_color[0][0][1] - 50, hsv_reference_color[0][0][2] - 50]
            print(f"Lower offset: {hsv_lower_offset}")
            hsv_upper_offset = [hsv_reference_color[0][0][0] + 10, hsv_reference_color[0][0][1] + 55, hsv_reference_color[0][0][2] + 50]
            print(f"Higher offset: {hsv_upper_offset}")
    # * Using live video
    else:
        hsv_lower_offset = [hsv_reference_color[0][0][0] - 10, hsv_reference_color[0][0][1] - 50, hsv_reference_color[0][0][2] - 50]
        hsv_upper_offset = [hsv_reference_color[0][0][0] + 10, hsv_reference_color[0][0][1] + 55, hsv_reference_color[0][0][2] + 50]

    return hsv_lower_offset, hsv_upper_offset

if __name__ == '__main__':

    # Testing with a static image
    test_with_static_img = True
    if test_with_static_img:
        #img_file_path = get_image_absolute_path('red_img_test.png')
        #mask_image_test(img_file_path)

        # Test with Husky static img
        img_file_path = get_image_absolute_path('husky.png')
        mask_image_test(img_file_path, test_with_static_img)


