"""
In OpenCV moments are the average of the intensities of an image's pixels

Segmentation is done before calculating the moments.
Moments is used to describe properties of an image, such:
- Centroid
- Area
- Orientation

Moments realtes the motion between two consecutive images, because
we can detect the difference between two images (what has been
unchanged).

In other words moments from an image are parameters to measure
the distribution of pixel intensities.
"""

import cv2
import numpy as np
from utils.cv_utils import define_hsv_limits_from_bgr, frame_rgb_to_hsv, mask_hsv_with_color


def get_two_largest_contours_and_merge(contours):
    # Calculates areas for contours
    contour_areas = [cv2.contourArea(contour) for contour in contours]

    # * Gets first largest area index
    first_index_of_largest_area_contour = [index for index, item in enumerate(contour_areas) if item == max(contour_areas)]
    first_index_of_largest_area_contour = first_index_of_largest_area_contour[0]

    # * Deletes recent found largest area so that the second largest can be computed
    del contour_areas[first_index_of_largest_area_contour]

    # * Gets second largest area index
    second_index_of_largest_area_contour = [index for index, item in enumerate(contour_areas) if item == max(contour_areas)]
    second_index_of_largest_area_contour = second_index_of_largest_area_contour[0]
    #print(f"Max contours are {contours[first_index_of_largest_area_contour[0]]} and {contours[second_index_of_largest_area_contour[0]+1]}")
    
    # Merged contour
    return np.vstack([contours[first_index_of_largest_area_contour], contours[second_index_of_largest_area_contour]])

def open_video_and_preprocess(video_file: str):
    print("Opening video")

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")

    hsv_lower, hsv_upper = define_hsv_limits_from_bgr([24, 22, 84], is_static_image=False)

    # Read until video is completed
    while (cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

        # Display the resulting frame
            hsv_frame = frame_rgb_to_hsv(frame)
            masked_frame = mask_hsv_with_color(hsv_frame, hsv_lower, hsv_upper)

            # We could extract just the V from HSV in order to get grayscale
            #masked_frame_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_HSV2BGR)
            masked_frame_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            _, masked_frame_binary = cv2.threshold(masked_frame_gray, 128, 255, cv2.THRESH_BINARY)
            cv2.imshow('binary', masked_frame_binary)
            
            contours, _ = cv2.findContours(image=masked_frame_binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            
            # * ------ Calculates two largest contours so that both can be concatenated into one ---------
            merged_contour = get_two_largest_contours_and_merge(contours)

            x, y, w, h = cv2.boundingRect(merged_contour)

            # TODO: Get orientation of the ellipse to get the angle of the object
            ellipse = cv2.fitEllipse(merged_contour)
            rect = cv2.minAreaRect(np.vstack(merged_contour))
            box = cv2.boxPoints(rect)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 4)
            cv2.ellipse(frame, ellipse, (0,0,255), 2)

            # draw contours on the original image
            image_copy = hsv_frame.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # see the results
            #cv2.imshow('None approximation', image_copy)
            #cv2.imwrite('contours_none_image1.jpg', image_copy)

            cv2.imshow('original_frame', frame)
            cv2.imshow('masked_frame', masked_frame)
            
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


if __name__ == '__main__':
    open_video_and_preprocess('../test_data/Video1_husky.mp4')