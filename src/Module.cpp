#include "pytorch.h"

torch::Module::Module()
{
	submodule_counter = 0;
}

torch::Module::~Module()
{

}

Tensor torch::Module::forward(Tensor input)
{
	return input;
}

string  torch::Module::tostring(int indentation_level)
{

	std::stringstream s;

	string indentation = string(indentation_level, ' ');

	s << indentation << module_name << " (" << std::endl;

	for (auto name_module_pair : modules)
	{

		s << indentation << " (" << name_module_pair.first << ") "
			<< name_module_pair.second->tostring(indentation_level + 1) << std::endl;
	}

	s << indentation << ")" << std::endl;

	return s.str();
}

void torch::Module::add_module(string module_name, Module::Ptr module)
{
	modules.push_back(pair<string, Module::Ptr>(module_name, module));
}

void torch::Module::add(Module::Ptr module)
{
	string module_name = std::to_string(submodule_counter);
	add_module(module_name, module);
	submodule_counter++;
}

map<string, Tensor> torch::Module::state_dict(map<string, Tensor> & destination, string prefix)
{
	// TODO: add another function that will not accept any parameters
	// and just return the state_dict()
	for (auto name_parameter_pair : parameters)
	{
		// Check if the parameter defined -- for example if we don't use bias
		// in the convolution, the bias weight will be undefined.
		// We need this in order to match the state_dict() function of Pytorch
		if (name_parameter_pair.second.defined())
		{
			destination[prefix + name_parameter_pair.first] = name_parameter_pair.second;
		}
	}

	for (auto name_buffer_pair : buffers)
	{
		destination[prefix + name_buffer_pair.first] = name_buffer_pair.second;
	}

	for (auto name_module_pair : modules)
	{
		name_module_pair.second->state_dict(destination, prefix + name_module_pair.first + '.');
	}

	return destination;
}

template<typename Func>	void torch::Module::apply(Func closure)
{
	for (auto name_parameter_pair : parameters)
	{
		if (name_parameter_pair.second.defined())
		{
			// maybe catch if it is undefined here
			parameters[name_parameter_pair.first] = closure(name_parameter_pair.second);
		}
	}

	for (auto name_buffer_pair : buffers)
	{
		buffers[name_buffer_pair.first] = closure(name_buffer_pair.second);
	}

	for (auto name_grad_pair : grads)
	{
		grads[name_grad_pair.first] = closure(name_grad_pair.second);
	}

	for (auto name_module_pair : modules)
	{
		name_module_pair.second->apply(closure);
	}
}

void torch::Module::cuda()
{
	// Transfer each tensor to GPU
	this->apply([](Tensor & tensor)
	{
		return tensor.toBackend(Backend::CUDA);
	}
	);
}

void torch::Module::cpu()
{
	// Transfer each tensor to CPU
	this->apply([](Tensor & tensor)
	{
		return tensor.toBackend(Backend::CPU);
	});
}

void torch::Module::save_weights(string hdf5_filename)
{
	map<string, Tensor> model_state_dict;
	this->state_dict(model_state_dict);
	save(hdf5_filename, model_state_dict);
}

void torch::Module::load_weights(string hdf5_filename)
{
	// TODO:
	// (1) Add check to make sure that the network is on cpu
	//     before loading weights
	// (2) Add support for not float. So far only works with
	//     float weights only.

	map<string, Tensor> model_state_dict;
	map<string, Tensor> checkpoint_dict;

	this->state_dict(model_state_dict);
	checkpoint_dict = load(hdf5_filename);

	// Compare model_state_dict -> checkpoint_dict keys consistency

	for (auto name_tensor_pair : model_state_dict)
	{
		if (checkpoint_dict.count(name_tensor_pair.first) != 1)
		{
			cout << "WARNING: model requires parameter ('" << name_tensor_pair.first << "') "
				<< "which is not present in the checkpoint file. Using model's default." << endl;
		}
	}

	// Compare checkpoint_dict -> model_state_dict keys consistency
	for (auto name_tensor_pair : checkpoint_dict)
	{
		if (model_state_dict.count(name_tensor_pair.first) != 1)
		{
			cout << "WARNING: checkpoint file contains parameter ('" << name_tensor_pair.first << "') "
				<< "which is not required by the model. The parameter is not used." << endl;
		}
	}

	for (auto name_tensor_pair : model_state_dict)
	{
		if (checkpoint_dict.count(name_tensor_pair.first) == 1)
		{
			// Copy in-place
			name_tensor_pair.second.copy_(checkpoint_dict[name_tensor_pair.first]);
		}
	}
}