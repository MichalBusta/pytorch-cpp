#include "pytorch.h"

torch::ReLU::ReLU()
{

};

torch::ReLU::~ReLU()
{

};

Tensor torch::ReLU::forward(Tensor input)
{
	Threshold_updateOutput(input, input, 0, 0, true);
	return input;
};


string torch::ReLU::tostring(int indentation_level)
{
	string indentation = string(indentation_level, ' ');

	return indentation + std::string("ReLU");
}