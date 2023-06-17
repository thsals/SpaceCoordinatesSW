#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

int main() {


	cv::VideoCapture vc(2);
	bool bCamRun = true;

	cv::Mat imgCap;
	cv::Mat frame, gray;
	cv::Mat cameraMatrix,distCoeffs,R,T;

	bool success;

	if(!vc.isOpened()) return -1;

	// Creating vector to store vectors of 3D points for each checkerboard image
	std::vector<std::vector<cv::Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	std::vector<std::vector<cv::Point2f> > imgpoints;

	// vector to store the pixel coordinates of detected checker board corners 
	std::vector<cv::Point2f> corner_pts;

	// Defining the world coordinates for 3D points
	std::vector<cv::Point3f> objp;
	for(int i{0}; i<CHECKERBOARD[1]; i++) {
		for(int j{0}; j<CHECKERBOARD[0]; j++)
			objp.push_back(cv::Point3f(j*0.02,i*0.02,0));
	}

	int seq = 0;

	while(bCamRun) {
		vc >> imgCap;
		if(imgCap.empty()) break;

		cv::imshow("origin", imgCap);
		char keyRtn = cv::waitKey(1);
		if(keyRtn == 27) break;
		if(keyRtn == 32) {
			cv::cvtColor(imgCap, gray, cv::COLOR_BGR2GRAY);
			// Finding checker board corners
			// If desired number of corners are found in the image then success = true  
			success = cv::findChessboardCorners(
					gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
					cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

			/* 
			 * If desired number of corner are detected,
			 * we refine the pixel coordinates and display 
			 * them on the images of checker board
			 */
			if(success) {
				printf("capture chessBoard\n");
				cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

				// refining pixel coordinates for given 2d points.
				cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

				// Displaying the detected corner points on the checker board
				cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

				//cv::imwrite("

				objpoints.push_back(objp);
				imgpoints.push_back(corner_pts);
			}
		}
	}


	/*
	 * Performing camera calibration by 
	 * passing the value of known 3D points (objpoints)
	 * and corresponding pixel coordinates of the 
	 * detected corners (imgpoints)
	 */
	cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);

	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distCoeffs : " << distCoeffs << std::endl;
	std::cout << "Rotation vector : " << R << std::endl;
	std::cout << "Translation vector : " << T << std::endl;




	cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
	//fs["camera_matrix"] >> intrinsic_matrix_loaded;
	//fs["distortion_coefficients"] >> distortion_coeffs_loaded;
	//cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
	//cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << "\n" << endl;

	// Build the undistort map which we will use for all
	// subsequent frames.
	//
	cv::Mat map1, map2;
	//cv::initUndistortRectifyMap(intrinsic_matrix_loaded, distortion_coeffs_loaded,
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs,
			cv::Mat(), cameraMatrix, imgCap.size(),
			CV_16SC2, map1, map2);


	cv::Mat imgRtn;
	cv::remap(imgCap, imgRtn, map1, map2, cv::INTER_LINEAR,
			cv::BORDER_CONSTANT, cv::Scalar());
	cv::imshow("Original", imgCap);
	cv::imshow("Undistorted", imgRtn);
	cv::waitKey(0);

	return 0;

	// Extracting path of individual image stored in a given directory
	std::vector<cv::String> images;
	// Path of the folder containing checkerboard images
	std::string path = "./images/*.jpg";

	cv::glob(path, images);

	//bool success;

	// Looping over all the images in the directory
	for(int i{0}; i<images.size(); i++)
	{
		frame = cv::imread(images[i]);
		cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true  
		success = cv::findChessboardCorners(
				gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts,
				cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		/* 
		 * If desired number of corner are detected,
		 * we refine the pixel coordinates and display 
		 * them on the images of checker board
		 */
		if(success)
		{
			cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);

			// Displaying the detected corner points on the checker board
			cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);
		}

		cv::imshow("Image",frame);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();


	/*
	 * Performing camera calibration by 
	 * passing the value of known 3D points (objpoints)
	 * and corresponding pixel coordinates of the 
	 * detected corners (imgpoints)
	 */
	cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);

	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distCoeffs : " << distCoeffs << std::endl;
	std::cout << "Rotation vector : " << R << std::endl;
	std::cout << "Translation vector : " << T << std::endl;

	return 0;
}
