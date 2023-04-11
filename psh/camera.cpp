#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

int main()
{

    Mat image;

    /*동영상 파일이나 카메라를 통해 들어오는 영상의 캡쳐를 위한 클래스.*/
    VideoCapture cap;

    /*VideoCapture클래스의 인스턴스를 통해서 연결된 웹캠의 제어를 받는 부분.0이라는 숫자는 카메라의 id값이다.
    현재 연결된 내부 카메라(built in webcam)를 의미한다.하지만 외부에 웹캠이 따로 연결되면 그 웹캠의 id값이 0이 된다.*/

    cap.open(0);

    namedWindow("window", CV_WINDOW_AUTOSIZE);

    /*루프를 돌면서 프레임 단위로 이미지를 받아들이는 부분.*/
    while (1)
    {
        /*VideoCapture로 이미지 프레임 하나를 받아서 image변수로 넘김.*/
        if (cap.read(image) == NULL)
        {
            cout << "frame was not read." << endl;
        }
        /*이미지 프레임을 윈도우를 통해서 스크린으로 출력.이 과정이 반복되면서 영상이 출력되게 된다.*/
        imshow("window", image);
        /*delay 33ms*/
        waitKey(33);
    }
    return 0;
}