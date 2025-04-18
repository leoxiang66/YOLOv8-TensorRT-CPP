#include "cmd_line_util.h"
#include "yolov8.h"

// Runs object detection on an input image then saves the annotated image to disk.
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

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV8.detectObjects(img);

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Print probability and rectangle for each object
    for (const auto& obj : objects) {
        std::cout << "Object: " << config.classNames[obj.label] << std::endl;
        std::cout << "Probability: " << obj.probability << std::endl;
        std::cout << "Rectangle: x=" << obj.rect.x 
                << ", y=" << obj.rect.y 
                << ", width=" << obj.rect.width 
                << ", height=" << obj.rect.height << std::endl;
    }

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}