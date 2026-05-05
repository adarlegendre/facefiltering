#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/core/core.hpp>        // basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // OpenCV image processing
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

/*---------------------------------------------------------------------------
TASK 1 
	!! Only add code to the reserved places.
	!! The resulting program must not print anything extra on any output (nothing more than the prepared program framework).
*/


/*	TASK 1.1 - Convert image from RGB to GRAYSCALE
	Iterate over pixels and convert RGB values to GRASCALE value.
	Convert final <float> value to <unsigned char> value.

	Allowed Mat attributes, methods and OpenCV functions for task solution are:
		Mat:: ... rows, cols, step(), size(), at<>(), zeros(), ones(), eye()

*/
int rgb2gray( Mat& bgr, Mat& gray )
{
	gray = cv::Mat::zeros( bgr.size(), CV_8UC1 );

	/* ***** Working area - begin ***** */

	
	/* ***** Working area - end ***** */

	return 0;
}


/* 	TASK 1.2 - Convolution 
	The input is a grayscale image (0-255) and a 3x3 float kernel (CV_32FC1).
    Leave the border values of the resulting image at 0.
	Implement manually by passing the image, for each pixel go through its surroundings and perform convolution with a 3x3 core.
    The resulting value must be normalized.

	Allowed Mat attributes, methods and OpenCV functions for task solution are:
		Mat:: ... rows, cols, step(), size(), at<>(), zeros(), ones(), eye()
    
*/
int convolution( cv::Mat& gray, const cv::Mat& kernel, cv::Mat& dst )
{
	dst = cv::Mat::zeros( gray.size(), CV_32FC1 );

	if( kernel.rows != 3 || kernel.cols != 3 )
		return 1;

	/* *****  Working area - begin ***** */

		/*
		iterate over 3x3 neighbourhood of the current point and:
			1. calculate the convolution of a local area with a convolution kernel
			2. normalize the result value
			3. save to the output image
		*/

	

	/* ***** Working area - end ***** */

	return 0;
}

//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// support functions
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save );
//---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		cout << "Not enough parameters" << endl;
		return -1;
	}

	// load image, use first argument, load as RGB image
	Mat img_rgb = imread(argv[1], IMREAD_COLOR);
	if(img_rgb.empty() )                              
	{
		cout <<  "Could not open or find the original image" << endl ;
		return 1;
	}

	//---------------------------------------------------------------------------
	// TASK: RGB convert to GRAYSCALE image
	
	Mat img_gray, img_ref;
	
	// compute solution using your own function
	rgb2gray(img_rgb, img_gray);
	
	// compute reference solution
	cvtColor(img_rgb, img_ref, cv::COLOR_RGB2GRAY);

	// compute and report differences
	checkDifferences( img_gray, img_ref, "rgb2gray", true );


	//---------------------------------------------------------------------------
	// TASK: convolution

	float ker[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	cv::Mat kernel( 3, 3, CV_32FC1, ker );
	cv::Mat conv_res, conv_ref;

	img_gray = img_ref;

	// compute solution using your own function
	convolution( img_gray, kernel, conv_res );

	// compute reference solution
	cv::flip( kernel, kernel, -1 );
	cv::filter2D( img_gray, conv_ref, CV_32F, kernel );
	// normalize convolution output
	conv_ref *= 1.f/(cv::sum(abs(kernel)).val[0] + 0.000000001);

	// since the filter2D function also calculates the values at the edges of the output image (and we don't for simplicity)
	// erase the edge values of the image before comparison
	cv::rectangle( conv_ref, cv::Point(0,0), cv::Point(conv_ref.cols-1,conv_ref.rows-1), cv::Scalar::all(0), 1 );

	// compute and report differences
	checkDifferences( conv_res, conv_ref, "convolution", true );

	//---------------------------------------------------------------------------

  
	return 0;
}
    



//---------------------------------------------------------------------------
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save )
{
	double mav = 255., err = 255., nonzeros = 1000.;

	if( !test.empty() && !ref.empty() ) {
		cv::Mat diff;
		cv::absdiff( test, ref, diff );
		cv::minMaxLoc( diff, NULL, &mav );
		nonzeros = 1. * cv::countNonZero( diff ); // / (diff.rows*diff.cols);
		err = (nonzeros > 0 ? ( cv::sum(diff).val[0] / nonzeros ) : 0);

		if( save ) {
			diff *= 255;
			cv::imwrite( (tag+".0.ref.png").c_str(), ref );
			cv::imwrite( (tag+".1.test.png").c_str(), test );
			cv::imwrite( (tag+".2.diff.png").c_str(), diff );
		}
	}

	printf( "%s_avg_cnt_max, %.1f, %.0f, %.0f, ", tag.c_str(), err, nonzeros, mav );
	
	return;
}
//---------------------------------------------------------------------------



