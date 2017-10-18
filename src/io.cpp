#include "pytorch.h"

map<string, Tensor> torch::load(string hdf5_filename)
{
	map<string, Tensor> tensor_dict;

	// use our get_names function
	vector<string> tensor_names = get_hdf5_file_keys(hdf5_filename);

	H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_RDONLY);

	// Array to store the shape of the current tensor
	hsize_t * dims_hsize_t;

	// We need this because one function can't accept hsize_t
	vector<int64_t> dims_int;

	// Float buffer to intermediately store weights
	float * float_buffer;

	// 'Rank' of the tensor
	int ndims;

	// Number of elements in the current tensor
	hsize_t tensor_flattened_size;

	Tensor buffer_tensor;

	for (auto tensor_name : tensor_names)
	{
		dims_int.clear();

		// Open a 'dataset' which stores current tensor
		H5::DataSet current_dataset = file.openDataSet(tensor_name);

		// We can infer the sizes of a store tensor from H5::DataSpace
		H5::DataSpace dataspace = current_dataset.getSpace();
		ndims = dataspace.getSimpleExtentNdims();

		// Get the overall number of elements -- we need this
		// to allocate the temporary buffer
		tensor_flattened_size = dataspace.getSimpleExtentNpoints();

		// Get the shame of the tensor
		dims_hsize_t = new hsize_t[ndims];
		dataspace.getSimpleExtentDims(dims_hsize_t, NULL);

		for (int i = 0; i < ndims; ++i)
		{
			// Converting hsize_t to int
			dims_int.push_back(int64_t(dims_hsize_t[i]));
		}

		// Allocate temporary float buffer
		// TODO: add support for other types like int
		// and make automatic type inference
		float_buffer = new float[tensor_flattened_size];

		current_dataset.read(float_buffer, H5::PredType::NATIVE_FLOAT,
			dataspace, dataspace);

		buffer_tensor = CPU(kFloat).tensorFromBlob(float_buffer, dims_int);

		tensor_dict[tensor_name] = buffer_tensor.type().copy(buffer_tensor);

		delete[] float_buffer;
		delete[] dims_hsize_t;
	}

	file.close();

	return tensor_dict;
}

void torch::save(string hdf5_filename, map<string, Tensor> dict_to_write)
{
	H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_TRUNC);

	for (auto name_tensor_pair : dict_to_write)
	{

		auto tensor_to_write = name_tensor_pair.second.contiguous();
		auto tensor_name = name_tensor_pair.first;

		auto dims = tensor_to_write.sizes();

		// The dimensionality of the tensor
		auto ndims = tensor_to_write.ndimension();
		auto tensor_flattened_size = tensor_to_write.numel();
		auto tensor_to_write_flatten = tensor_to_write.view({ -1 });
		auto tensor_to_write_flatten_accessor = tensor_to_write_flatten.accessor<float, 1>();

		float * float_buffer = new float[tensor_flattened_size];

		// Convert an array of ints into an array of hsize_t
		auto dims_hsize_t = new hsize_t[ndims];

		for (int i = 0; i < ndims; ++i)
		{
			dims_hsize_t[i] = dims[i];
		}

		for (int i = 0; i < tensor_flattened_size; ++i)
		{

			float_buffer[i] = tensor_to_write_flatten_accessor[i];
		}

		H5::DataSpace space(ndims, dims_hsize_t);

		H5::DataSet dataset = H5::DataSet(file.createDataSet(tensor_name,
			H5::PredType::NATIVE_FLOAT,
			space));


		dataset.write(float_buffer, H5::PredType::NATIVE_FLOAT);

		delete[] float_buffer;
	}

	file.close();
}

vector<string> torch::get_hdf5_file_keys(string hdf5_filename)
{
	// We open and close hdf5 file here. It might be an overkill
	// as we can open the file once, read keys and read tensors outright,
	// but this way we also add a simple debugging function to be able to
	// easily get keys without dealing with HDF5 API directly.

	// Open the file
	H5::H5File file = H5::H5File(hdf5_filename, H5F_ACC_RDONLY);

	vector<string> names;

	// Define a closure to populate our names array
	auto closure = [](hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
	{

		vector<string> * names_array_pointer = reinterpret_cast< vector<string> *>(opdata);

		names_array_pointer->push_back(string(name));

		return 0;
	};

	// Run our closure and populate array
	H5Literate(file.getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, closure, &names);

	file.close();

	return names;
}