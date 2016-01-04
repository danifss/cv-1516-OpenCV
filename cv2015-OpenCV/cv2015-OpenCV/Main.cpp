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
Mat imgOrig; // Original image
Mat canny_img, canny_gray; // Canny
Mat canny_dst, detected_edges; // Canny
Mat sobel_grad, sobel_img, sobel_gray; // Sobel
Mat robert_grad, robert_img, robert_gray; // Robert
Mat prewitt_grad, prewitt_img, prewitt_gray; // Prewitt
Mat freichen_grad, freichen_img, freichen_gray; // Frei Chen
Mat marrhildreth_img, marrhildreth_gray, marrhildreth_dst; // marr-hildreth

int edgeThresh = 1;
int threshold_value_c = 0;
int threshold_value_mh = 0;
int lowThreshold = 0;
int const max_lowThreshold = 255;
int ratio = 3;
int kernel_size = 3;
string windowName[] = { "Original Image", "Canny Edge Detector", "Sobel Edge Detector", "Robert's Edge Detector",
						"Prewitt Edge Detector", "Frei Chen Detector", "Marr-Hildreth Edge Detector" };

/// Functions
void CannyThreshold(int, void*);
void SobelDerivatives();
void RobertsDetector();
void PrewittDetector();
void FreiChenDetector();
void MarrHildrethDetector(int, void*);

// Auxiliar functions
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
	cout << "\tFrei Chen Edge Detector" << endl;
	cout << "\tMarr-Hildreth Edge Detector" << endl;
	
	// Read image name
	string imgName;
	cout << endl << "Choose an image name:(see img folder) " << endl;
	cout << "[ lena.jpg, deti.jpg, sta2.bmp, wdg2.bmp, tools_2.png, mon1.bmp, ape1.bmp Bikesgray.jpg ]" << endl;
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
	canny_dst.create(imgOrig.size(), imgOrig.type());
	canny_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(canny_img, canny_gray, COLOR_BGR2GRAY);
	namedWindow(windowName[1], WINDOW_AUTOSIZE); // Window
	// Create a Trackbar for user to enter threshold
	createTrackbar("Canny Threshold:", windowName[1], &threshold_value_c, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);

	cout << endl << "Canny Image:";
	printImageFeatures(canny_dst); // Show information about image


	// --------------- Sobel Edge Detector ----------------
	sobel_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(sobel_img, sobel_gray, COLOR_BGR2GRAY);
	// Call function
	SobelDerivatives();

	cout << endl << "Sobel Image:";
	printImageFeatures(sobel_grad); // Show information about image


	// ------------- Robert's Edge Detector ---------------
	robert_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(robert_img, robert_gray, COLOR_BGR2GRAY);
	// Call function
	RobertsDetector();

	cout << endl << "Robert's Image:";
	printImageFeatures(robert_grad); // Show information about image


	// -------------- Prewitt Edge Detector ----------------
	prewitt_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(prewitt_img, prewitt_gray, COLOR_BGR2GRAY);
	// Call function
	PrewittDetector();

	cout << endl << "Prewitt Image:";
	printImageFeatures(prewitt_grad); // Show information about image


	// ------------- Frei Chen Edge Detector ---------------
	freichen_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(freichen_img, freichen_gray, COLOR_BGR2GRAY);
	// Call function
	FreiChenDetector();

	cout << endl << "Frei Chen Image:";
	printImageFeatures(freichen_grad); // Show information about image


	// ------------- Marr-Hildreth Edge Detector ---------------
	marrhildreth_img = imgOrig.clone();
	// Convert the image to grayscale
	cvtColor(marrhildreth_img, marrhildreth_gray, COLOR_BGR2GRAY);
	namedWindow(windowName[6], WINDOW_AUTOSIZE); // Window
	// Create Trackbar
	createTrackbar("Marr-Hildreth Threshold:", windowName[6], &threshold_value_mh, max_lowThreshold, MarrHildrethDetector);
	MarrHildrethDetector(0, 0);

	cout << endl << "Frei Chen Image:";
	printImageFeatures(marrhildreth_dst); // Show information about image


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
	Canny(detected_edges, detected_edges, lowThreshold, threshold_value_c*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	canny_dst = Scalar::all(0);

	canny_img.copyTo(canny_dst, detected_edges);
	imshow(windowName[1], canny_dst);
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

	GaussianBlur(sobel_gray, sobel_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
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
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_grad);

	namedWindow(windowName[2], WINDOW_AUTOSIZE); // Window
	imshow(windowName[2], sobel_grad);

	/// OTHER VERSION
/*
	Mat sobelx;
	Sobel(sobel_gray, sobelx, CV_32F, 1, 0);

	double minVal, maxVal;
	minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
	cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

	Mat draw;
	sobelx.convertTo(draw, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

	namedWindow("image", CV_WINDOW_AUTOSIZE);
	imshow("image", draw);
*/
}


/**
* @function RobertsDetector
* @brief Trackbar callback - Applies a convolution, then its like Sobel, applying gradiant.
*/
void RobertsDetector() {

	GaussianBlur(robert_gray, robert_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	Matx22f matx = Matx22f(1, 0, 0, -1);
	Matx22f maty = Matx22f(0, 1, -1, 0);
	InputArray kernelx = _InputArray(matx);
	InputArray kernely = _InputArray(maty);
	

	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	filter2D(robert_gray, grad_x, -1, kernelx, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	filter2D(robert_gray, grad_y, -1, kernely, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, robert_grad);

	namedWindow(windowName[3], WINDOW_AUTOSIZE); // Window
	imshow(windowName[3], robert_grad);
}


/**
* @function PrewittDetector
* @brief Trackbar callback - Applies a different convolution of robert's, then its like Sobel, applying gradiant.
*/
void PrewittDetector() {

	GaussianBlur(prewitt_gray, prewitt_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Matx33f matx = Matx33f(-1, 0, 1, -1, 0, 1, -1, 0, 1);
	Matx33f maty = Matx33f(-1, -1, -1, 0, 0, 0, 1, 1, 1);
	InputArray kernelx = _InputArray(matx);
	InputArray kernely = _InputArray(maty);


	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	filter2D(prewitt_gray, grad_x, -1, kernelx, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	filter2D(prewitt_gray, grad_y, -1, kernely, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, prewitt_grad);

	namedWindow(windowName[4], WINDOW_AUTOSIZE); // Window
	imshow(windowName[4], prewitt_grad);
}


/**
* @function FreiChenDetector
* @brief Trackbar callback - Applies four convolutions to image
*/
void FreiChenDetector() {

	GaussianBlur(freichen_gray, freichen_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Matx33f mat1 = Matx33f(2, 3, 4, 0, 0, 0, -2, -3, -2);
	Matx33f mat2 = Matx33f(2, 0, -2, 0, -2, 3, 3, -2, 0);
	Matx33f mat3 = Matx33f(3, 0, -3, 2, 0, -2, -2, 0, 2);
	Matx33f mat4 = Matx33f(2, 0, -2, -3, 2, 0, 0, 2, -3);
	InputArray kernel1 = _InputArray(mat1);
	InputArray kernel2 = _InputArray(mat2);
	InputArray kernel3 = _InputArray(mat3);
	InputArray kernel4 = _InputArray(mat4);


	// Generate grad_1, ...
	Mat grad_1, grad_2, grad_3, grad_4;
	Mat abs_grad_1, abs_grad_2, abs_grad_3, abs_grad_4;

	/// Gradient 1
	filter2D(freichen_gray, grad_1, -1, kernel1, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_1, abs_grad_1);

	/// Gradient 2
	filter2D(freichen_gray, grad_2, -1, kernel2, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_2, abs_grad_2);

	/// Gradient 3
	filter2D(freichen_gray, grad_3, -1, kernel3, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_3, abs_grad_3);

	/// Gradient 4
	filter2D(freichen_gray, grad_4, -1, kernel4, Point(-1, -1), 0, BORDER_DEFAULT);
	convertScaleAbs(grad_4, abs_grad_4);

	/// Total Gradient (approximate)
	Mat gradient_tmp1, gradient_tmp2, gradient_tmp3;
	addWeighted(abs_grad_1, 0.5, abs_grad_2, 0.5, 0, gradient_tmp1);
	addWeighted(abs_grad_3, 0.5, abs_grad_4, 0.5, 0, gradient_tmp2);
	addWeighted(gradient_tmp1, 0.5, gradient_tmp2, 0.5, 0, freichen_grad);

	namedWindow(windowName[5], WINDOW_AUTOSIZE); // Window
	imshow(windowName[5], freichen_grad);
}


/**
* @function MarrHildrethDetector
*/
void MarrHildrethDetector(int, void*) {
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	GaussianBlur(marrhildreth_gray, marrhildreth_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

	/// Apply Laplace function
	Mat abs_dst, laplace_dst, thres_dst, tmp_dst;

	Laplacian(marrhildreth_gray, laplace_dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(laplace_dst, abs_dst);

	/// Apply Threshold
	threshold(abs_dst, thres_dst, threshold_value_mh*ratio, max_lowThreshold, 3);

	/// Using Marr-Hildreth's output as a mask, we display our result
	marrhildreth_dst = Scalar::all(0);
	marrhildreth_img.copyTo(marrhildreth_dst, thres_dst);

	imshow(windowName[6], marrhildreth_dst);
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
