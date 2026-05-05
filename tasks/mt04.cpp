#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <opencv2/core/core.hpp>        // basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // OpenCV image processing
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

void spectrumMag( const cv::Mat& src, cv::Mat& mag);

/*---------------------------------------------------------------------------
TASK 4
	!! Only add code to the reserved places.
	!! The resulting program must not print anything extra on any output (nothing more than the prepared program framework).
*/

// Support functions 
// Compare test and reference images and report the differences.
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save = false);


/*	TASK 4.0 - Peak signal-to-noise ratio

	copy and understand the opencv tutorial
	reference:	http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html#image-similarity-psnr-and-ssim
	
*/
double getPSNR(const Mat& I1, const Mat& I2)
{
	double psnr = 0.0;
	

	/* ***** Working area - begin ***** */
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2];

	if (sse <= 1e-10) // for small values return zero
		return 0.0;

	double mse = sse / (double)(I1.channels() * I1.total());
	psnr = 10.0 * log10((255 * 255) / mse);
	/* ***** Working area - end ***** */

 	return psnr; 
}



/*	TASK 4.1 - Salt&Pepper Noise
	Implement a function for adding Salt&Pepper Noise to the image.
	Reset appropriate amount of invalid pixels according to probability.
	Set these pixels randomly to 0 or 255.
	You do not need to solve the reseting of one pixel twice.

*/

void noiseSaltAndPepper( const Mat& src, Mat& dst, double probability )
{
	dst = src.clone();

	/* ***** Working area - begin ***** */
	Mat sp_noise = Mat::zeros(src.size(), CV_8U);
	randu(sp_noise, 0, 255);

	int limit_approx = floor(probability * 128);

	Mat black = sp_noise < limit_approx;
	Mat white = sp_noise > 256 - limit_approx;

	dst.setTo(255, white);
	dst.setTo(0, black);
	/* ***** Working area - end ***** */
}


/* Reference solution.
*/

void noiseSaltAndPepperRef(const Mat& src, Mat& dst, double probability)
{
	Mat sp_noise = Mat::zeros(src.size(),CV_8U);
	randu(sp_noise,0,255);

	int limit_approx = floor(probability*128);

	Mat black = sp_noise < limit_approx;
	Mat white = sp_noise > 256-limit_approx;
	
	dst = src.clone();
	dst.setTo(255,white);
	dst.setTo(0,black);
	return;
}


/*	TASK 4.2 - Gaussian Noise
	Implement a function for adding Gaussian Noise to the image.
	Change pixel values based on random values with normal distribution.
*/
void noiseGaussian( const Mat& src, Mat& dst, double stddev, double mean = 0.0 )
{
	cv::RNG rng(0xFFFFFFFF);
	dst = src.clone();

	/* ***** Working area - begin ***** */
	Mat noise = Mat(src.size(), CV_64F);
	Mat r1 = Mat(src.size(), CV_64F);
	src.convertTo(r1, CV_64F);
	randn(noise, 0, stddev);
	r1 = r1 + mean + 255. * noise;
	r1.convertTo(dst, src.type());
	/* ***** Working area - end ***** */

}


/* Reference solution.
*/
void noiseGaussianRef(const Mat& src, Mat& dst, double stddev, double mean = 0.0)
{
	
	Mat noise = Mat(src.size(),CV_64F);
	Mat r1    = Mat(src.size(),CV_64F);
	src.convertTo(r1, CV_64F);
	randn(noise, 0, stddev);
	r1 = r1 + mean + 255.*noise;
	r1.convertTo(dst, src.type());
	return;
}



/*--------------------------------------------------------------------------- */

/* 	Examples of input parameters
	mt04 image_path noise_type param_1 [param_2 = 0.0]

	- noise type
	  sp - saltr&pepper noise, param_1 is a probability of the noise
	  gn - Gaussian noise, param_1, resp. param_2 is a standard deviation, resp. mean
*/

int main(int argc, char* argv[])
{
    std::string img_path = "";
    std::string noise_type = "";
    double p1, p2 = 0;

	if (argc < 4)
	{
		cout << "Not enough parameters." << endl;
		cout << "Usage: mt04 image_path noise_type param_1 [param_2 = 0.0]" << endl;
		return -1;
	}

	// check input parameters
	if( argc > 1 ) img_path = std::string( argv[1] );
	if( argc > 2 ) noise_type = std::string( argv[2] );
	if( argc > 3 ) p1 = atof( argv[3] );
	if( argc > 4 ) p2 = atof( argv[4] );

	// load testing images
	cv::Mat rgb = cv::imread( img_path );

	// check testing images
	if( rgb.empty() ) {
		std::cout << "Failed to load image: " << img_path << std::endl;
		return -1;
	}

	cv::Mat gray;
	cv::cvtColor( rgb, gray, CV_BGR2GRAY );

	//---------------------------------------------------------------------------

	Mat test, ref;

	if (noise_type == "sp")
	{
		Mat test, ref;
		noiseSaltAndPepper( gray, test, p1 );
		noiseSaltAndPepperRef( gray, ref, p1 );
		checkDifferences( test, ref, "sp", true );	// to store output images 
		std::cout << ", PSNR(test), " << getPSNR( gray, test );
		std::cout << ", PSNR(ref), " << getPSNR( gray, ref ) << std::endl;
	}
	else if (noise_type == "gn")
	{
		noiseGaussian( gray, test, p1, p2 );
		noiseGaussianRef( gray, ref, p1, p2 );
		checkDifferences( test, ref, "gn", true );	// to store output images 
		std::cout << ", PSNR(test), " << getPSNR( gray, test );
		std::cout << ", PSNR(ref), " << getPSNR( gray, ref ) << std::endl;
	}

    return 0;
}
//---------------------------------------------------------------------------




void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save )
{
	double mav = 255., err = 255., nonzeros = 1000.;

	if( !test.empty() ) {
		cv::Mat diff;
		cv::absdiff( test, ref, diff );
		cv::minMaxLoc( diff, NULL, &mav );
		nonzeros = 1. * cv::countNonZero( diff ); // / (diff.rows*diff.cols);
		err = (nonzeros > 0 ? ( cv::sum(diff).val[0] / nonzeros ) : 0);
		nonzeros /= (diff.rows*diff.cols);

		if( save ) {
			diff *= 255;
			cv::imwrite( (tag+".0.ref.png").c_str(), ref );
			cv::imwrite( (tag+".1.test.png").c_str(), test );
			cv::imwrite( (tag+".2.diff.png").c_str(), diff );
		}
	}

	printf( "%s, avg, %.1f, perc, %2.2f, max, %.0f, ", tag.c_str(), err, 100.*nonzeros, mav );

	return;
}



