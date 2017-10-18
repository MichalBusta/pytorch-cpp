#include "pytorch.h"

torch::Sequential::Sequential()
{
	module_name = "Sequential";
};

torch::Sequential::~Sequential()
{

};

// Forward for sequential block makes forward pass
// for each submodule and passed it to the next one
Tensor torch::Sequential::forward(Tensor input)
{
	Tensor out = input;

	for (auto name_module_pair : modules)
	{
		out = name_module_pair.second->forward(out);
	}

	return out;
}

torch::Module::Ptr torch::Sequential::get(int i) const
{
	return modules[i].second;
}