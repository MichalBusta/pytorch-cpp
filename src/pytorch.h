#ifndef PYTORCH_H
#define PYTORCH_H

#include "ATen/ATen.h"
#include "ATen/Type.h"

#include <sstream>
#include <map>
#include "H5Cpp.h"

#include <opencv2/opencv.hpp>

#define TENSOR_DEFAULT_TYPE CPU(kFloat)

using namespace at;


using std::map;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;
using std::tie;

using namespace cv;

namespace torch
{
	map<string, Tensor> load(string hdf5_filename);
	void save(string hdf5_filename, map<string, Tensor> dict_to_write);

	class Module
	{
	public:


		// Sequential module needs the counter
		// as names of submodules are not provided
		// sometimes.
		int submodule_counter;
		string module_name = "Module";

		Module();
		~Module();

		// We will use pointer to other modules a lot
		// This is done to automatically handle deallocation of created
		// module objects
		typedef shared_ptr<Module> Ptr;
		virtual Tensor Forward(Tensor input);

		// This function gets overwritten
		// for the leafnodes like Conv2d, AvgPool2d and so on
		virtual string ToString(int indentation_level = 0);

		// vector<pair<string, Ptr>> because we want to emulate
		// the ordered dict this way, meaning that elements
		// are stored in the same order they were added

		// Like in Pytorch each module stores the modules that it uses
		vector<pair<string, Ptr>> modules;

		// And parameters that are explicitly used by the current module
		map<string, Tensor> parameters;

		// Plus buffers which are meant to store running mean and var for batchnorm layers
		map<string, Tensor> buffers;

		// We store parameter related to gradient computation here and other
		// tensors so far
		// TODO: some members of grads are not related to gradient computation
		//       and were put there temporary -- put them in a more relevant container.
		map<string, Tensor> grads;

		// A function to add another modules inside current module
		// Acts as Pytorch's Module.add_module() function
		void add_module(string module_name, Module::Ptr module);

		// Sometimes, when modules are being added, not all of them
		// have weights, like RELU. In this case the weights can be
		// numerated out of order. For example:
		// net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
		// net.state_dict().keys()
		// output: ['0.weight', '0.bias', '2.weight', '2.bias']

		// Equivalent behaviour will be seen with the add() function
		// described below: if relu is added, the counter for weights will
		// be increased.
		void add(Module::Ptr module);

		map<string, Tensor> state_dict(map<string, Tensor> & destination, string prefix = "");
		template<typename Func>	void apply(Func closure);
		void cuda();
		void cpu();
		void save_weights(string hdf5_filename);
	};
}

#endif // !PYTORCH_H
