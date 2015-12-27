/*
 * OpenCV project for CV
 * Multiple algorithms for edge detecting.
 * 
 * Daniel Silva - 51908
 * João Cravo - 6
 */

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

// Functions
void printImageFeatures(const Mat &imagem);


// MAIN PROGRAM
int main( void ) /// int argc, char** argv
{
	// Select algorithm
	string op;
	cout << "Edge detector algorithms:" << endl;
	cout << "\t1 - Canny Edge Detector" << endl;
	cout << "\t2 - Marr-Hildreth Edge Detector" << endl;
	cout << "\t3 - Robert's Edge Detector" << endl;
	cout << "\t4 - Prewitt Edge Detector" << endl;
	cout << "\t5 - Sobel Edge Detector" << endl;
	cout << "\t6 - Frie Chen Edge Detector" << endl;
	cout << endl << "Choose number of the algorithm: ";
	cin >> op;
	


	Mat imgOrig;
	imgOrig = imread("img/wdg2.bmp", CV_LOAD_IMAGE_UNCHANGED);

	if (!imgOrig.data) {
		// Leitura SEM SUCESSO
		cout << "Ficheiro nao foi aberto ou localizado !!" << endl;
		return -1;
	}
	if (imgOrig.channels() > 1) {
		// Converter para 1 so canal !!
		cvtColor(imgOrig, imgOrig, CV_BGR2GRAY, 1);
	}
	// Janela
	namedWindow("Imagem Original", CV_WINDOW_AUTOSIZE);
	// Visualizar
	imshow("Imagem Original", imgOrig);
	// Imprimir alguma informacao
	cout << "IMAGEM ORIGINAL" << endl;
	printImageFeatures(imgOrig);


	/// FINAL ---------------------------------------------
	// Wait
	waitKey(0);
	// Destroy all windows
	destroyAllWindows();

	return 0;
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