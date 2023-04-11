#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main() {

	VideoCapture cap(0);
	Mat img;

	while (1) {
		cap >> img;

		imshow("camera img", img);

		if (waitKey(1) == 27) break;
	}


	return 0;
}