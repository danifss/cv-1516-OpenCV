/*
 * OpenCV project for CV
 * Multiple algorithms for edge detecting.
 * 
 * Daniel Silva - 51908
 * João Cravo - 63784
 */

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

/// Global variables
Mat imgOrig, imgOrig_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 255;
int ratio = 3;
int kernel_size = 3;
string windowName = "Original Image";

/// Functions
void CannyThreshold(int, void*);
void printImageFeatures(const Mat &imagem);


/// MAIN PROGRAM
int main( void ) /// int argc, char** argv
{
	// Select algorithm
	cout << "Edge detector algorithms supported:" << endl;
	cout << "\tCanny Edge Detector" << endl;
	cout << "\tSobel Edge Detector" << endl;
	cout << "\tRobert's Edge Detector" << endl;
	cout << "\tPrewitt Edge Detector" << endl;
	cout << "\tFrie Chen Edge Detector" << endl;
	cout << "\tMarr-Hildreth Edge Detector" << endl;
	
	// Read image name
	string imgName;
	cout << endl << "Choose an image name:(see img folder) " << endl;
	cout << "[ lena.jpg, deti.jpg, sta2.bmp, wdg2.bmp, tools_2.png, mon1.bmp, ape1.bmp ]" << endl;
	cout << endl << "Photo name: ";
	cin >> imgName;

	// Read original image
	imgOrig = imread("img/"+imgName);

	if (imgOrig.empty()) {
		// Error reading image
		cout << "Ficheiro nao foi aberto ou localizado !!" << endl;
		return -1;
	}
	namedWindow(windowName, WINDOW_AUTOSIZE); // Window
	imshow(windowName, imgOrig); // Show image
	cout << endl << "Original Image:";
	printImageFeatures(imgOrig); // Show information about image


	// --------------- Canny Edge Detector ----------------
	// Create a matrix of the same type and size as src (for dst)
	dst.create(imgOrig.size(), imgOrig.type());
	// Convert the image to grayscale
	cvtColor(imgOrig, imgOrig_gray, COLOR_BGR2GRAY);
	windowName = "Canny Edge Detector";
	namedWindow(windowName, WINDOW_AUTOSIZE); // Window
	// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", windowName, &lowThreshold, max_lowThreshold, CannyThreshold);
	// Show the image
	CannyThreshold(0, 0);


	/// FINAL ---------------------------------------------
	// Wait
	waitKey(0);
	// Destroy all windows
	destroyAllWindows();

	return 0;
}


/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/
void CannyThreshold(int, void*) {
	/// Reduce noise with a kernel 3x3
	blur(imgOrig_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	imgOrig.copyTo(dst, detected_edges);
	imshow(windowName, dst);
}


// FUNCOES AUXILIARES
void printImageFeatures(const Mat &imagem)
{
	cout << endl;
	cout << "Numero de linhas : " << imagem.size().height << endl;
	cout << "Numero de colunas : " << imagem.size().width << endl;
	cout << "Numero de canais : " << imagem.channels() << endl;
	cout << "Numero de bytes por pixel : " << imagem.elemSize() << endl;
	cout << endl;
}
