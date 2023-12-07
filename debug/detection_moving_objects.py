import cv2
import os
import random
import numpy as np

folder_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_videos/'
# folder_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/debug/ray_videos'

close_window = False

# Set a threshold for motion detection
motion_threshold = 25


def motion_mask(frame, prev_gray):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between the current and previous frame
    frame_diff = cv2.absdiff(prev_gray, gray)

    # Apply a threshold to get the motion areas
    _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

    # Find contours of motion areas
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around motion areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Update the previous frame
    prev_gray = gray.copy()

    return frame, prev_gray


def gaussian(frame, bg_subtractor):
    # Apply Gaussian Mixture-based Background/Foreground Segmentation to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise and improve the mask
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, (5, 5))

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around moving objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return frame


def optical_flow(frame, prev_gray):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract motion vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold to get motion areas
    motion_mask = np.zeros_like(prev_gray)
    motion_mask[magnitude > 10] = 255

    # Find contours of motion areas
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around motion areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    prev_gray = gray.copy()

    return frame, prev_gray


def threshold(frame, bg_subtractor):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtain the background mask
    mask = bg_subtractor.apply(gray)

    # Apply a threshold to the mask
    _, thresh = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around moving objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return frame


if __name__ == '__main__':
    # Open the video
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    random.shuffle(video_files)

    # Create a resizable window
    cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Motion Detection', 800, 600)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        # Initialize the first frame
        ret, frame = cap.read()
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame, prev_gray = motion_mask(frame, prev_gray)
            # frame = gaussian(frame, bg_subtractor)
            # optical_flow(frame, prev_gray)
            # threshold(frame, bg_subtractor)

            # Show the video with bounding boxes
            cv2.imshow('Motion Detection', frame)
            cv2.waitKey(50)

            # Close Window
            if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
                close_window = True
                break

        if close_window:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
