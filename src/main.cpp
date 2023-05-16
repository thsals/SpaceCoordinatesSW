#include <fstream>

#include <yolo_v2_class.hpp>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

//std::map<unsigned int, std::string> objNames;
std::vector<std::string> classes;

void divideImage(const cv::Mat &imgSrc, cv::Mat &matL, cv::Mat &matR, bool setGray = true) {
	cv::Mat imgGray;
	if(setGray && imgSrc.channels() == 3) cv::cvtColor(imgSrc, imgGray, cv::COLOR_BGR2GRAY);
	else imgGray = imgSrc.clone();

	int cols = imgGray.cols/2;
	matL = imgGray(cv::Rect(0, 0, cols, imgGray.rows)).clone();
	matR = imgGray(cv::Rect(cols, 0, cols, imgGray.rows)).clone();
}

void findMatching(const cv::Mat &roi, const cv::Mat &origin) {

	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	detector->detectAndCompute( roi, cv::noArray(), keypoints1, descriptors1 );
	detector->detectAndCompute( origin, cv::noArray(), keypoints2, descriptors2 );

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SURF is a floating-point descriptor NORM_L2 is used
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<cv::DMatch> > knn_matches;
	matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
	const float ratio_thresh = 0.7f;
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++) {
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
			good_matches.push_back(knn_matches[i][0]);
		}
	}


//	const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
//	const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
//
//	std::vector<cv::KeyPoint> kpts1, kpts2;
//	cv::Mat desc1, desc2;
//
//	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
//	cv::BFMatcher matcher(cv::NORM_HAMMING);
//	std::vector< std::vector<cv::DMatch> > nn_matches;
//	matcher.knnMatch(desc1, desc2, nn_matches, 2);
//	std::vector<cv::KeyPoint> matched1, matched2;
//	for(size_t i = 0; i < nn_matches.size(); i++) {
//		cv::DMatch first = nn_matches[i][0];
//		float dist1 = nn_matches[i][0].distance;
//		float dist2 = nn_matches[i][1].distance;
//		if(dist1 < nn_match_ratio * dist2) {
//			matched1.push_back(kpts1[first.queryIdx]);
//			matched2.push_back(kpts2[first.trainIdx]);
//		}
//	}
//	std::vector<cv::DMatch> good_matches;
//	std::vector<cv::KeyPoint> inliers1, inliers2;
//	for(size_t i = 0; i < matched1.size(); i++) {
//		cv::Mat col = cv::Mat::ones(3, 1, CV_64F);
//		col.at<double>(0) = matched1[i].pt.x;
//		col.at<double>(1) = matched1[i].pt.y;
//		col = homography * col;
//		col /= col.at<double>(2);
//		double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
//				pow(col.at<double>(1) - matched2[i].pt.y, 2));
//		if(dist < inlier_threshold) {
//			int new_i = static_cast<int>(inliers1.size());
//			inliers1.push_back(matched1[i]);
//			inliers2.push_back(matched2[i]);
//			good_matches.push_back(cv::DMatch(new_i, new_i, 0));
//		}
//	}

	//-- Draw matches
	cv::Mat img_matches;
	cv::drawMatches( roi, keypoints1, origin, keypoints2, good_matches, img_matches,
			cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected matches
	cv::imshow("Good Matches", img_matches );
	cv::waitKey();
}

bool doProcess(Detector *pYolo, const cv::Mat &img) {
	//cv::Mat imgDraw = img.clone();
	cv::Mat imgL, imgR;

	divideImage(img, imgL, imgR, false);

	std::vector<bbox_t> boxes;
	boxes = pYolo->detect(imgL);

	//printf("Yolo result [%d]\n", boxes.size());
	cv::Mat roi;
	for(bbox_t box : boxes) {
		if(box.obj_id == 41) {
			cv::rectangle(imgL, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar::all(0));
			//printf("[%d : %s] detected\n", box.obj_id, classes[box.obj_id].c_str());
			roi = img(cv::Rect(box.x, box.y, box.w, box.h));
			findMatching(roi, imgR);
		}
	}
	cv::imshow("yolo", imgL);
	return true;
}

void testSingleImage(Detector *pYolo, const char *imgPathName) {
	cv::Mat img = cv::imread(imgPathName);

	doProcess(pYolo, img);

	cv::imshow("logo", img);
	cv::waitKey(0);
}

void doCamLoop(Detector *pDetector, int camNum) {
	cv::Mat img;
	cv::VideoCapture vc(camNum);
		
	while(vc.isOpened()) {
		vc >> img;
		if(!doProcess(pDetector, img)) vc.release();

		cv::imshow("logo", img);
		if(cv::waitKey(1)==27) vc.release();
	}
}

bool fileToMap(const std::string &filename) {
	std::ifstream ifile;
	ifile.open(filename.c_str());
	if(!ifile) return false;   //could not read the file.
	classes.clear();

	std::string sName;
	unsigned int key = 0;
	while( ifile >> sName ) {
		//nameMap[key]= sName;
		//key++;
		classes.push_back(sName);
	}
	return classes.empty()?false:true;
}

int main(int argv, char** argc) {
	//Detector detector("../../darknet/cfg/yolov4.cfg", "../../darknet/yolov4.weights");
	Detector detector("../../darknet/cfg/yolov4.cfg", "../../darknet/yolov4.weights");
	fileToMap("../../darknet/data/coco.names");

	if(argv == 1) {
		//testSingleImage(&detector, "../opencvLogo.png");
		//testSingleImage(&detector, "../../darknet/data/dog.jpg");
		testSingleImage(&detector, "../deskSample.jpg");
	} else {
		doCamLoop(&detector, atoi(argc[1]));
	}
	return 0;
}
