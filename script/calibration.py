import cv2 as cv
import os
import numpy as np

# Checkerboard size
CHESS_BOARD_DIM = (9, 7)

# The size of Square in the checkerboard.
SQUARE_SIZE = 19  # millimeters

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3D points in real-world space
img_points_2D = []  # 2D points in image plane.

# Lists to store calibration results from all images
all_cam_matrices = []
all_dist_coeffs = []
all_rvecs = []
all_tvecs = []
all_img_points = []

# The images directory path
image_dir_path = "images"

files = os.listdir(image_dir_path)

for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    print(f"Loading image: {imagePath}")

    # Load the image
    image = cv.imread(imagePath)

    # Check if the image loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{imagePath}'. Skipping.")
        continue

    # Convert to grayscale
    grayScale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply preprocessing steps if necessary (e.g., Gaussian blur, thresholding)
    grayBlur = cv.GaussianBlur(grayScale, (5, 5), 0)
    ret, thresh = cv.threshold(grayBlur, 100, 255, cv.THRESH_BINARY)
    
    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(thresh, CHESS_BOARD_DIM, None)
    
    if ret == True:
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
        obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
        obj_3D[:, :2] = np.mgrid[0:CHESS_BOARD_DIM[0], 0:CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
        obj_3D *= SQUARE_SIZE
        obj_points_3D.append(obj_3D)
        
        # Refine corner positions
        corners2 = cv.cornerSubPix(grayScale, corners, (11, 11), (-1, -1), criteria)
        img_points_2D.append(corners2)
        all_img_points.append(corners2)

        # Draw corners on the image (for visualization)
        img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)

        # Show the image with corners detected (for debugging)
        cv.imshow('Chessboard Image', img)
        cv.waitKey(500)  # Show image for 500 milliseconds

        # Perform camera calibration for this image
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            [obj_3D], [corners2], grayScale.shape[::-1], None, None)

        # Collect calibration results
        all_cam_matrices.append(mtx)
        all_dist_coeffs.append(dist)
        all_rvecs.extend(rvecs)
        all_tvecs.extend(tvecs)

cv.destroyAllWindows()

# Perform averaging of calibration results if at least one image was successfully processed
if len(all_cam_matrices) > 0:
    # Calculate average calibration parameters
    avg_cam_matrix = np.mean(all_cam_matrices, axis=0)
    avg_dist_coeffs = np.mean(all_dist_coeffs, axis=0)
    avg_rvecs = np.mean(all_rvecs, axis=0)
    avg_tvecs = np.mean(all_tvecs, axis=0)

    # Print averaged results
    print("Average Camera Matrix (mtx):\n", avg_cam_matrix)
    print("Average Distortion Coefficients (dist):\n", avg_dist_coeffs)
    print("Average Rotation Vectors (rvecs):\n", avg_rvecs)
    print("Average Translation Vectors (tvecs):\n", avg_tvecs)

    # Calculate mean reprojection error
    mean_error = 0
    for i in range(len(obj_points_3D)):
        imgpoints2, _ = cv.projectPoints(obj_points_3D[i], all_rvecs[i], all_tvecs[i], avg_cam_matrix, avg_dist_coeffs)
        error = cv.norm(all_img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Mean Reprojection Error: {}".format(mean_error / len(obj_points_3D)))

else:
    print("No chessboard corners found in any images.")
