#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int ac, char** av) {

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("카메라를 열 수 없음");
		return 0;
	}

	Mat img;

	while (1)
	{	
		
		cap >> img;

		imshow("카메라", img);
		if (waitKey(1) == 27)
			break;
	}
	return 0;
}