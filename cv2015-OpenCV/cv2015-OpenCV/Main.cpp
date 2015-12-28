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
Mat imgOrig;
Mat canny_img, canny_gray;
Mat dst, detected_edges;
Mat grad, sobel_img, sobel_gray;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 255;
int ratio = 3;
int kernel_size = 3;
string windowName[] = { "Original Image", "Canny Edge Detector", "Sobel Edge Detector" };

/// Functions
void CannyThreshold(int, void*);
void SobelDerivatives();
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
	namedWindow(windowName[0], WINDOW_AUTOSIZE); // Window
	imshow(windowName[0], imgOrig); // Show image
	cout << endl << "Original Image:";
	printImageFeatures(imgOrig); // Show information about image


	// --------------- Canny Edge Detector ----------------
	// Create a matrix of the same type and size as src (for dst)
	dst.create(imgOrig.size(), imgOrig.type());
	canny_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(canny_img, canny_gray, COLOR_BGR2GRAY);
	namedWindow(windowName[1], WINDOW_AUTOSIZE); // Window
	// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", windowName[1], &lowThreshold, max_lowThreshold, CannyThreshold);
	// Show the image
	CannyThreshold(0, 0);

	cout << endl << "Canny Image:";
	printImageFeatures(dst); // Show information about image


	// --------------- Sobel Edge Detector ----------------
	sobel_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(imgOrig, sobel_gray, COLOR_BGR2GRAY);
	// Call function
	SobelDerivatives();

	cout << endl << "Sobel Image:";
	printImageFeatures(grad); // Show information about image



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
	blur(canny_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	canny_img.copyTo(dst, detected_edges);
	imshow(windowName[1], dst);
}


/**
* @function SobelDerivatives
* @brief Trackbar callback - Applies the Sobel Operator and generates as output an 
*                            image with the detected edges bright on a darker background.
*/
void SobelDerivatives() {
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	GaussianBlur(sobel_img, sobel_img, Size(3, 3), 0, 0, BORDER_DEFAULT);

	namedWindow(windowName[2], WINDOW_AUTOSIZE); // Window
	
	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	
	/// Gradient X
	Sobel(sobel_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	
	/// Gradient Y
	Sobel(sobel_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	imshow(windowName[2], grad);
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
