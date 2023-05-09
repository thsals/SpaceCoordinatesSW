#include <yolo_v2_class.hpp>


bool doProcess(Detector *pYolo, const cv::Mat &img) {

	cv::Mat imgDraw = img.clone();
	std::vector<bbox_t> boxes;

	boxes = pYolo->detect(img);

	//printf("Yolo result [%d]\n", boxes.size());
	for(bbox_t box : boxes) {
		cv::rectangle(imgDraw, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar::all(0));
	}
	cv::imshow("yolo", imgDraw);
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

int main(int argv, char** argc) {
	//Detector detector("../../darknet/cfg/yolov4.cfg", "../../darknet/yolov4.weights");
	Detector detector("../../darknet/cfg/yolov4-tiny.cfg", "../../darknet/yolov4-tiny.weights");
	if(argv == 1) {
		//testSingleImage(&detector, "../opencvLogo.png");
		testSingleImage(&detector, "../../darknet/data/dog.jpg");
	} else {
		doCamLoop(&detector, atoi(argc[1]));
	}
	return 0;
}
