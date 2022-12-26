window.SIDEBAR_ITEMS = {"fn":[["base64_image_to_image","decodes base64 encoded image to dynamic image"],["generate_random_subimages","generate_random_subimages() takes in an image, a count, and two dimensions (width and height) as parameters and returns a vector of DynamicImages. It creates a ThreadRng to generate random numbers between the width/2 and the image’s width, and between the height/2 and the image’s height. It then uses window_crop() to crop the image at those coordinates with the given dimensions, and pushes it into the vector of DynamicImages."],["image_to_base64_image","converts dynamic image to base64 encoded png image"],["keypoint_windows","This function calculates keypoints based on windows from an image. It takes in a DynamicImage, the number of keypoints to be calculated (count), and the size of each window (window_size). It first creates an empty vector to store the windows. It then gets the width and height of the image, and calculates the center of each window. The image is then blurred with a blur factor of 2.5, and corners_fast9 is used to calculate keypoints from the blurred image. The keypoints are sorted by score, and then iterated over until count is reached. For each keypoint, its x and y coordinates are found along with its window size, and pushed into the vector of windows. Finally, this vector is returned."],["pyramid","This function calculates a pyramid of windows from an image. The scale, step size, and window size are all arguments that can be passed in. The image is resized to the width and height divided by the scale. A sliding window is then created with the step size and window size arguments. The windows are then mapped to their original coordinates multiplied by the scale and collected into a vector of Windows."],["rotated_frames","rotated_frames() takes in a DynamicImage and returns an iterator over the rotated windows of the given image. It uses rotate_about_center() to rotate the image by a given radian angle, and uses Nearest interpolation with a black background. The radian angles used are 0.02, -0.02, 0.05, -0.05, 0.07, -0.07, 0.09, -0.09, 1.1, -1.1, 1.3, -1.3, 1.5, -1.5 and 2.0,-2.0"],["scaled_frames","scaled_frames() takes in a DynamicImage and returns an iterator over scaled frames of the given image. The iterator contains four frames, with scaling factors of 0.8, 0.9, 1.1, and 1.2 respectively. The warp() function is used to scale the image using nearest neighbor interpolation and a black background color (Rgb([0, 0, 0]))."],["sliding_window","This function calculates a sliding window based on an image, a step size and a window size. It returns a vector of windows, which are composed of the x and y coordinates of the top left corner of the window, as well as the DynamicImage associated with that window."],["window_crop","window_crop() takes in an image, a window width and height, and a center point (x, y) as parameters. It then crops the image to the size of the window width and height, with the center point being the center of the cropped image. It returns a DynamicImage type with the cropped image."]]};