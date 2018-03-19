#include "pytorch.h"

torch::ReLU::ReLU()
{

};

torch::ReLU::~ReLU()
{

};

Tensor torch::ReLU::forward(Tensor input)
{
	//threshold_forward_out(input, input, 0, 0);
	return input.clamp_min(0);
};


string torch::ReLU::tostring(int indentation_level)
{
	string indentation = string(indentation_level, ' ');

	return indentation + std::string("ReLU");
}
