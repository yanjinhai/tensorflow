#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/interpreter.h>

constexpr int IMAGE_WIDTH = 28;
constexpr int IMAGE_HEIGHT = 28;
constexpr int BLOCK_RADIUS = 2;

void writePixel(std::vector<float>& image, int x, int y) {
	if (0 <= x && x < IMAGE_WIDTH && 0 <= y && y < IMAGE_HEIGHT) {
		int index = y * IMAGE_WIDTH + x;
		image[index] = 1.0;
	}
}

std::vector<float> generateImage(int x, int y, int radius) {
	std::vector<float> image(IMAGE_WIDTH * IMAGE_HEIGHT, 0.0);
	for (int i = x - radius; i <= x + radius; i++) {
		for (int j = y - radius; j <= y + radius; j++) {
			writePixel(image, i, j);
		}
	}
	return image;
}

int main() {
	std::cout << "Generating images...\n";

	std::vector<std::vector<float>> images;
	std::vector<std::pair<int, int>> labels;
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			images.push_back(generateImage(j, i, BLOCK_RADIUS));
			labels.emplace_back(j, i);
		}
	}

	std::cout << "Done.\n";

	// Load the model
	std::string modelPath = "converted_model.tflite";
	std::unique_ptr<tflite::FlatBufferModel> model;
	model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
	if (!model) {
		std::cerr << "Failed to load the model: " << modelPath << std::endl;
		return 1;
	}

	// Create interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	if (!interpreter) {
		std::cerr << "Failed to create interpreter." << std::endl;
		return 1;
	}

	interpreter->AllocateTensors();

	// std::cout << "1a. Tensors size: " << interpreter->tensors_size() << std::endl;
	// std::cout << "1b. Inputs size: " << interpreter->inputs().size() << std::endl;
	// std::cout << "1c. Outputs size: " << interpreter->outputs().size() << std::endl;

	// Get input and output tensors
	int inputTensorIndex = interpreter->inputs()[0];
	TfLiteTensor* inputTensor = interpreter->tensor(inputTensorIndex);
	int outputTensorIndex = interpreter->outputs()[0];
	TfLiteTensor* outputTensor = interpreter->tensor(outputTensorIndex);

	// std::cout << "2a. Tensors size: " << interpreter->tensors_size() << std::endl;
	// std::cout << "2b. Inputs size: " << interpreter->inputs().size() << std::endl;
	// std::cout << "2c. Outputs size: " << interpreter->outputs().size() << std::endl;

	// std::cout << "3a. Input tensor index: " << interpreter->inputs()[0] << std::endl;
	// std::cout << "3b. Output tensor index: " << interpreter->outputs()[0] << std::endl;

	// Run inference on the input images
	for (size_t i = 0; i < images.size(); i++) {
		// Set input tensor data
		std::memcpy(inputTensor->data.f, images[i].data(),
		      IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));

		// Run the interpreter
		interpreter->Invoke();

		// Access the prediction result
		float xPred = outputTensor->data.f[1];
		float yPred = outputTensor->data.f[0];

		// Calculate the distance between predicted and true values
		float distance = std::sqrt(std::pow(labels[i].first - xPred, 2)
			     + std::pow(labels[i].second - yPred, 2));

		// Print the prediction results and distance
		std::cout << xPred << " " << yPred << " | " << labels[i].first
			<< " " << labels[i].second << " | " << distance <<
		std::endl;
	}

	return 0;
}

