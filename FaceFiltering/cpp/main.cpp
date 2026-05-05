/**
 * Minimal C++ OpenCV CLI: Sobel magnitude or median on an image.
 * Usage: face_filter_cli <input_path> <output_path> <sobel|median> [ksize]
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <input> <output> <sobel|median> [ksize]\n";
		return 1;
	}
	std::string inPath = argv[1];
	std::string outPath = argv[2];
	std::string mode = argv[3];
	int ksize = (argc > 4) ? std::atoi(argv[4]) : 3;
	if (ksize % 2 == 0) ++ksize;
	if (ksize < 3) ksize = 3;

	cv::Mat bgr = cv::imread(inPath, cv::IMREAD_COLOR);
	if (bgr.empty()) {
		std::cerr << "Failed to read: " << inPath << "\n";
		return 2;
	}

	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::Mat out;

	if (mode == "sobel") {
		cv::Mat gx, gy;
		cv::Sobel(gray, gx, CV_32F, 1, 0, ksize);
		cv::Sobel(gray, gy, CV_32F, 0, 1, ksize);
		cv::magnitude(gx, gy, out);
		cv::normalize(out, out, 0, 255, cv::NORM_MINMAX);
		out.convertTo(out, CV_8U);
		cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
	} else if (mode == "median") {
		cv::medianBlur(bgr, out, ksize);
	} else {
		std::cerr << "Unknown mode: " << mode << "\n";
		return 3;
	}

	if (!cv::imwrite(outPath, out)) {
		std::cerr << "Failed to write: " << outPath << "\n";
		return 4;
	}
	std::cout << "Wrote " << outPath << "\n";
	return 0;
}
