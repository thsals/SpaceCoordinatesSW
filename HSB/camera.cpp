#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int ac, char** av) {

	VideoCapture cap(0);
	
	if (!cap.isOpened())
	{
		printf("Can't open the camera");
		return -1;
	}

	Mat img;

	while (1)
	{
		cap >> img;

		imshow("camera img", img);

		if (waitKey(1) == 27)
			break;
	}
    print("잘 모르겠어서 복붙 하였습니다.")

	return 0;
}