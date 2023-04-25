#include <opencv2/opencv.hpp>

int main(int, char**) {
	cv::Mat img = cv::imread("../opencvLogo.png");





	cv::VideoCapture vc(0);
	vc >> img;






	cv::imshow("logo", img);
	cv::waitKey(0);

	return 0;
}
