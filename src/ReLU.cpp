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

torch::CReLU::CReLU()
{

};

torch::CReLU::~CReLU()
{

};

Tensor torch::CReLU::forward(Tensor input)
{
	//threshold_forward_out(input, input, 0, 0);
	auto tmp = - input;
	auto concat =  cat({input, tmp}, 1);
	return concat.clamp_min(0);
};


string torch::CReLU::tostring(int indentation_level)
{
	string indentation = string(indentation_level, ' ');

	return indentation + std::string("CReLU");
}
