import cv2
import numpy as np

from executables.approximate_policy_decision_trees import load_partial_data

if __name__ == '__main__':
    # Assuming you have an array containing 100 images with a shape of (100, 84, 84, 4)
    # Replace this with your own array of images
    observations, actions = load_partial_data(
        '/home/malaarabiou/Programming_Projects/Pycharm_Projects/RLlib_Pong/ray_dataset/data_1.h5',
        0,
        1000,
    )
    images_array = observations
    images_array = (images_array * 255).astype(np.uint8)
    # images_array = np.random.randint(0, 255, size=(100, 84, 84, 4), dtype=np.uint8)  # Random example
    # Define video dimensions based on the shape of the images
    height, width, _ = images_array[0].shape

    # Create a VideoWriter object to write the video
    video_name = 'output_video.mp4'
    fps = 10  # Frames per second in the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video compression format (MP4)
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Add each image to the video
    for i in range(len(images_array)):
        # Convert the image to a format compatible with OpenCV (BGR)
        image_bgr = cv2.cvtColor(images_array[i], cv2.COLOR_RGBA2BGR)

        # Write the image into the video
        out.write(image_bgr)

    # Release the VideoWriter resource and display a message once finished
    out.release()
    print(f"The video '{video_name}' has been created successfully!")