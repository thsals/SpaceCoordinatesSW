#include <opencv2/opencv.hpp>

int main(int, char**) {
	cv::Mat img = cv::imread("../opencvLogo.png");

	cv::imshow("logo", img);
	cv::waitKey(0);

	return 0;
}
