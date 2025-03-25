#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputImage;

    // Parse the command line arguments
    if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
        return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);

    std::string inputDir = "/home/git/YOLOv8-TensorRT-CPP/evaluation/coco128/images/train2017";
    std::string outputDir = "/home/git/YOLOv8-TensorRT-CPP/evaluation/coco128/predicted";

    // Ensure output directory exists
    fs::create_directories(outputDir);

    // Process each file in the directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            std::string inputImagePath = entry.path().string();
            auto img = cv::imread(inputImagePath);
            if (img.empty()) {
                std::cerr << "Error: Unable to read image at path '" << inputImagePath << "'" << std::endl;
                continue;
            }

            // Get image dimensions
            int imgWidth = img.cols;
            int imgHeight = img.rows;

            // Run inference
            const auto objects = yoloV8.detectObjects(img);

            // Prepare output path
            std::string baseFilename = entry.path().stem().string();
            std::string outputPath = outputDir + "/" + baseFilename + ".txt";

            // Open file for writing results
            std::ofstream outputFile(outputPath);
            if (!outputFile.is_open()) {
                std::cerr << "Failed to open file for writing: " << outputPath << std::endl;
                continue;
            }

            // Write detection results to file
            for (const auto& obj : objects) {
                // Normalize the coordinates and dimensions
                float normX = obj.rect.x / (float)imgWidth;
                float normY = obj.rect.y / (float)imgHeight;
                float normWidth = obj.rect.width / (float)imgWidth;
                float normHeight = obj.rect.height / (float)imgHeight;

                outputFile << obj.label << " "
                           << normX << " "
                           << normY << " "
                           << normWidth << " "
                           << normHeight << "\n";
            }

            outputFile.close();
            std::cout << "Results saved to: " << outputPath << std::endl;
        }
    }

    return 0;
}