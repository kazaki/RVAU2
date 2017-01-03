#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <windows.h>
#include <sys/stat.h>
#include <sys/types.h> 
#include <map>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>


using namespace cv;
using namespace std;

static double markerLength = 0.05;

void drawCube(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
	InputArray _rvec, InputArray _tvec, float length) {

	float x2 = length * 2;

	// project axis points
	vector< Point3f > cubePoints;
	cubePoints.push_back(Point3f(-length, -length, 0));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(-length, -length, 0));

	cubePoints.push_back(Point3f(-length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, x2));
	cubePoints.push_back(Point3f(-length, -length, x2));

	cubePoints.push_back(Point3f(-length, -length, 0));
	cubePoints.push_back(Point3f(-length, -length, x2));
	cubePoints.push_back(Point3f(-length, length, 0));
	cubePoints.push_back(Point3f(-length, length, x2));
	cubePoints.push_back(Point3f(length, length, 0));
	cubePoints.push_back(Point3f(length, length, x2));
	cubePoints.push_back(Point3f(length, -length, 0));
	cubePoints.push_back(Point3f(length, -length, x2));

	cubePoints.push_back(Point3f(0, markerLength, 0));
	cubePoints.push_back(Point3f(0, 0, markerLength));
	vector< Point2f > imagePoints;
	projectPoints(cubePoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

	// draw axis lines

	for (int i=0; i < imagePoints.size() - 2; i+=2)
		line(_image, imagePoints[i], imagePoints[i+1], Scalar(0, 0, 255), 3);
	
}

void drawPyramid(InputOutputArray _image, InputArray _cameraMatrix, InputArray _distCoeffs,
	InputArray _rvec, InputArray _tvec, float length) {

	float x2 = length * 2;

	// project axis points
	vector< Point3f > pyrPoints;
	pyrPoints.push_back(Point3f(-length, -length, 0));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(-length, -length, 0));

	pyrPoints.push_back(Point3f(-length, -length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(-length, length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(length, length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));
	pyrPoints.push_back(Point3f(length, -length, 0));
	pyrPoints.push_back(Point3f(0, 0, x2));

	pyrPoints.push_back(Point3f(0, markerLength, 0));
	pyrPoints.push_back(Point3f(0, 0, markerLength));
	vector< Point2f > imagePoints;
	projectPoints(pyrPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

	// draw axis lines

	for (int i = 0; i < imagePoints.size() - 2; i += 2)
		line(_image, imagePoints[i], imagePoints[i + 1], Scalar(0, 255, 0), 3);

}

int main(int argc, char* argv[]) {

	cout << "Showing markers..." << endl;

	HANDLE dir;
	WIN32_FIND_DATA file_data;

	if ((dir = FindFirstFile("markers/*", &file_data)) == INVALID_HANDLE_VALUE) {
		cout << "No markers found." << endl;
		return 1;
	}

	vector<string> markers;
	Mat markerImg;

	// loop threw markers
	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = "markers/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		Mat markerImg = imread(full_file_name);
		imshow(file_name, markerImg);

		markers.push_back(file_name);

	} while (FindNextFile(dir, &file_data));
	waitKey(0);


	// pick what object each marker represents
	int ans;
	vector<int> answers;
	for (unsigned i = 0; i < markers.size(); i++) {
		markers.at(i);
		cout << "What would you like to assign " + markers.at(i) + " with? (1:cube 2:pyramid) ";
		cin >> ans;
		answers.push_back(ans);
	}

	if (answers.size() != markers.size()) {
		cout << "An error has occured" << endl;
		return 1;
	}




	// capture video feed and detect markers
	cv::VideoCapture inputVideo;
	inputVideo.open(0);

	// read camera parameters
	Mat cameraMatrix, distCoeffs;
	FileStorage fs("camera_parameters.txt", FileStorage::READ);
	fs["ip"] >> cameraMatrix;
	fs["dc"] >> distCoeffs;

	const cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

	while (inputVideo.grab()) {
		cv::Mat image, imageCopy;
		inputVideo.retrieve(image);
		image.copyTo(imageCopy);
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f> > corners;
		cv::aruco::detectMarkers(image, dictionary, corners, ids);
		// if at least one marker detected
		if (ids.size() > 0) {
			//cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
			vector< Vec3d > rvecs, tvecs;
			cv::aruco::estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

			for (int i = 0; i < ids.size(); i++) {
				if (ids[i] == 23)
				drawCube(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.025);
				else
				drawPyramid(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.025);
				//cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
			}


		}
		cv::imshow("out", imageCopy);
		char key = (char)cv::waitKey(100);
		if (key == 27)
			break;

	}


	waitKey(0);




	system("PAUSE");
	return 0;
}