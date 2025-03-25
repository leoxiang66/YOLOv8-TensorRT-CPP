cd build

./benchmark --model ../yolov8x.onnx --input ../images/640_640.jpg 

./benchmark --model ../yolov8x.onnx --input ../images/640_640.jpg --precision INT8 --calibration-data ../val2017

./benchmark2 --model ../yolov8x.onnx --input ../images/640_640.jpg 

./benchmark2 --model ../yolov8x.onnx --input ../images/640_640.jpg --precision INT8 --calibration-data ../val2017


./benchmark2 --model ../yolov8n.onnx --input ../images/640_640.jpg 

./benchmark2 --model ../yolov8n.onnx --input ../images/640_640.jpg --precision INT8 --calibration-data ../val2017