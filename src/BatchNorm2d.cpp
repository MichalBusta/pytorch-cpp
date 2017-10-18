#include "pytorch.h"

torch::BatchNorm2d::BatchNorm2d(
	int num_features,
	double eps,
	double momentum,
	bool affine,
	bool training) :
	num_features(num_features),
	eps(eps),
	momentum(momentum),
	affine(affine),
	training(training)
{
	// Initialize weights here

	// Ones initialization is temporarry -- just to avoid
	// division by zero during testing
	parameters["weight"] = TENSOR_DEFAULT_TYPE.ones(num_features);
	parameters["bias"] = TENSOR_DEFAULT_TYPE.zeros(num_features);

	buffers["running_mean"] = TENSOR_DEFAULT_TYPE.zeros(num_features);
	buffers["running_var"] = TENSOR_DEFAULT_TYPE.ones(num_features);

	// We don't recompute the mean and var during inference
	// So, some variables are initialized for possible future use case.
	grads["save_mean"] = TENSOR_DEFAULT_TYPE.ones(num_features);
	grads["save_std"] = TENSOR_DEFAULT_TYPE.ones(num_features);

};

torch::BatchNorm2d::~BatchNorm2d()
{

};

string torch::BatchNorm2d::tostring(int indentation_level)
{

	std::stringstream string_stream;

	string indentation = string(indentation_level, ' ');

	string_stream << indentation
		<< "BatchNorm2d( "
		<< "num_features=" << std::to_string(num_features) << " "
		<< "eps=" << std::to_string(eps) << " "
		<< "momentum=" << std::to_string(momentum) << " )";

	return string_stream.str();

};

Tensor torch::BatchNorm2d::forward(Tensor input)
{

	Tensor output = input.type().tensor();

	BatchNormalization_updateOutput(
		input,
		output,
		parameters["weight"],
		parameters["bias"],
		buffers["running_mean"],
		buffers["running_var"],
		grads["save_mean"],
		grads["save_std"],
		training,
		momentum,
		eps);
	return output;
};