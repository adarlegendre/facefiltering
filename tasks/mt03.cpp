#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <opencv2/core/core.hpp>        // basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // OpenCV image processing
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

// OpenCV 4 drops the old CV_* names from default headers; map them so the rest of the skeleton

#ifndef CV_MINMAX
#define CV_MINMAX cv::NORM_MINMAX
#endif
#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#endif

using namespace std;
using namespace cv;

#define LOW_PASS_FILTER 1
#define HIGH_PASS_FILTER 2

/*---------------------------------------------------------------------------
TASK 3
	!! Only add code to the reserved places.
	!! The resulting program must not print anything extra on any output (nothing more than the prepared program framework).
*/

// Support functions 

// The function rearranges the quadrants of the spectrum so that the origin is in the middle of the spcetrum.
void rearrangeSpectrum( cv::Mat& s );

// The function calculates the amplitude spectrum - absolute values from complex numbers in pixels. 
cv::Mat spectrumMagnitude( cv::Mat & specCplx );

// Compare test and reference images and report the differences.
void checkDifferences( const cv::Mat test, const cv::Mat ref, std::string tag, bool save = false);



/*	TASK 3.1 - High/Low pass filter in frequency domain

	Implement a function for filtering the image in the spectral domain.
	Implement high/low pass filters (controlled by function parameter flag) by setting the particular 
	spectrum parts to 0, thus, for example, for the high pass filter, reset the square area around the 
	center of the spectrum of size limit_frequency.		

	The Discrete Fourier Tranform is used to compute the image spectrum.

*/
int passFilter( const cv::Mat& src, cv::Mat& dst, int limit_frequency, int flag, cv::Mat * spektrum = NULL )
{
	if( src.empty())
		return 1;

	// find the optimal image size for efficient DFT calculation, use cv::getOptimalDFTSize ()
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(src.cols);
	dftSize.height = cv::getOptimalDFTSize(src.rows);

	// prepare a new image of the optimal size, set the values outside the source image to 0
	cv::Mat srcPadded;
	srcPadded = cv::Mat::zeros(dftSize, CV_32FC1);
	src.convertTo(srcPadded(cv::Rect(0,0, src.cols, src.rows)), srcPadded.type());
    
	// use the DFT function of the OpenCV library to calculate the spectrum
	cv::Mat spectrumCplx;
	cv::dft(srcPadded, spectrumCplx, cv::DFT_COMPLEX_OUTPUT, src.rows );

	// rearrange the quadrants of the spectrum so that the zero frequencies are in the center of the image
	rearrangeSpectrum( spectrumCplx );


	/* ***** Working area - begin ***** */

	// After rearrangeSpectrum(), the zero-frequency (DC) component sits near the geometric centre of
	// the spectrum image. Nearby coefficients correspond to slowly changing brightness in the
	// spatial image (low spatial frequencies); coefficients far from the centre correspond to fine
	// detail and edges (high spatial frequencies).
	//
	// We implement simple  brick-wall filters by forcing unwanted Fourier coefficients to
	// exactly zero. Each pixel of spectrumCplx is a complex number (two float channels: real and
	// imaginary). Setting both to zero removes that frequency from the reconstruction; partial
	// zeroing would break the inverse DFT in a non-physical way.
	//
	// The suggested text refers to a square neighbourhood around the centre of size
	// limit_frequency. We interpret that as an axis-aligned square of exactly
	// limit_frequency × limit_frequency bins centred on (cx, cy), using the usual integer split
	// for even sizes (for n = 20 you get dx ∈ [−10, 9] and the same for dy).

	// Centre of the shifted spectrum (matches rearrangeSpectrum's quadrant split at cols/2, rows/2).
	const int cx = spectrumCplx.cols / 2;
	const int cy = spectrumCplx.rows / 2;
	// Treat limit_frequency as the edge length of the kept (low-pass) square, not the half-width.
	// For an n×n block centred on (cx, cy), use n = limit_frequency with integer split n = neg + 1 + pos.
	const int n = limit_frequency;
	const int neg = n / 2;
	const int pos = ( n - 1 ) - neg;

	for( int y = 0; y < spectrumCplx.rows; y++ )
	{
		cv::Vec2f* row = spectrumCplx.ptr<cv::Vec2f>( y );
		for( int x = 0; x < spectrumCplx.cols; x++ )
		{
			const int dx = x - cx;
			const int dy = y - cy;
			// Axis-aligned square of size limit_frequency × limit_frequency around DC.
			const bool insideLowFreqBlock = ( dx >= -neg && dx <= pos && dy >= -neg && dy <= pos );

			if( flag == HIGH_PASS_FILTER )
			{
				// High-pass: we want only frequencies *above* the limit — remove the low-frequency
				// square around DC so that sharp structures survive and smooth shading is suppressed.
				if( insideLowFreqBlock )
					row[x] = cv::Vec2f( 0.f, 0.f );
			}
			else if( flag == LOW_PASS_FILTER )
			{
				// Low-pass: keep only frequencies *below* the limit — zero everything outside the
				// central square so the image becomes smoother (detail is cut off).
				if( !insideLowFreqBlock )
					row[x] = cv::Vec2f( 0.f, 0.f );
			}
		}
	}

	/* ***** Working area - end ***** */


	// export of the spectrum for evaluation purposes
	if( spektrum != NULL ) *spektrum = spectrumCplx.clone();
		
	// rearrange the quadrants of the spectrum back
	rearrangeSpectrum( spectrumCplx );

	// use the inverse dft (idft) function of the OpenCV library for the inverse transformation
	cv::dft( spectrumCplx, srcPadded, cv::DFT_REAL_OUTPUT+cv::DFT_INVERSE+cv::DFT_SCALE, src.rows );

	// normalize the output values and convert them to the original image format
	cv::normalize( srcPadded, srcPadded, 0, 1, CV_MINMAX );
	srcPadded(cv::Rect(0,0, src.cols, src.rows)).convertTo( dst, src.type(), 255, 0 );

	return 0;
}





/* 	Examples of input parameters
	mt03 image_path spatial_frequency_limit [path to reference results]

	- spatial_frequency_limit for testing with reference results is 20
*/
int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		cout << "Not enough parameters." << endl;
		cout << "Usage: mt03 image_path spatial_frequency_limit [path to reference results]" << endl;
		return -1;
	}

    std::string img_path = "";
    std::string ref_path = "";
    int limit_frequency = 100;

	// check input parameters
	if( argc > 1 ) img_path = std::string( argv[1] );
	if( argc > 2 ) limit_frequency = atoi( argv[2] );
	if( argc > 3 ) ref_path = std::string( argv[3] );

	// load testing images
	cv::Mat src_rgb = cv::imread( img_path );

	// check testing images
	if( src_rgb.empty() ) 
	{
		cout <<  "Could not open the original image: " << img_path << endl ;
		return 1;
	}
	cv::Mat src_gray;
	cv::cvtColor( src_rgb, src_gray, CV_BGR2GRAY );

	//---------------------------------------------------------------------------

	cv::Mat low_pass, high_pass;
	cv::Mat low_spec_cplx, high_spec_cplx;
	cv::Mat low_spec, high_spec;
	passFilter( src_gray, low_pass,  limit_frequency, LOW_PASS_FILTER,  &low_spec_cplx );
	passFilter( src_gray, high_pass, limit_frequency, HIGH_PASS_FILTER, &high_spec_cplx );
	spectrumMagnitude(low_spec_cplx).convertTo(low_spec, CV_8UC1, 255 );
	spectrumMagnitude(high_spec_cplx).convertTo(high_spec, CV_8UC1, 255 ) ;

	cv::Mat low_pass_ref, high_pass_ref;
	cv::Mat low_spec_ref, high_spec_ref;
	low_pass_ref  = cv::imread( ref_path+"low_pass_ref.png",  cv::IMREAD_GRAYSCALE );
	high_pass_ref = cv::imread( ref_path+"high_pass_ref.png", cv::IMREAD_GRAYSCALE );
	low_spec_ref  = cv::imread( ref_path+"low_spec_ref.png",  cv::IMREAD_GRAYSCALE );
	high_spec_ref = cv::imread( ref_path+"high_spec_ref.png", cv::IMREAD_GRAYSCALE );
	if( low_pass_ref.empty() || low_spec_ref.empty() || high_pass_ref.empty() || high_spec_ref.empty() ) 
		std::cout << "WARNING: references data failed to load." << std::endl;
	
	// vyhodnocení
	checkDifferences( low_pass, low_pass_ref,   "LPI", true );
	checkDifferences( low_spec, low_spec_ref,   "LPS", true );
	checkDifferences( high_pass, high_pass_ref, "HPI", true );
	checkDifferences( high_spec, high_spec_ref, "HPS", true );
	std::cout << std::endl;

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



void rearrangeSpectrum( cv::Mat& s )
{
    int cx = s.cols/2;
    int cy = s.rows/2;

    cv::Mat q0(s, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(s, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(s, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(s, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

cv::Mat spectrumMagnitude( cv::Mat & specCplx )
{
	cv::Mat specMag, planes[2];
	cv::split(specCplx, planes);						
	cv::magnitude(planes[0], planes[1], planes[0]);		
	cv::log( (planes[0] + cv::Scalar::all(1)), specMag );
	cv::normalize( specMag, specMag, 0, 1, CV_MINMAX );
	return specMag;
}

