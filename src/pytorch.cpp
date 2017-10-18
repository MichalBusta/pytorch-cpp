#include "pytorch.h"

Tensor torch::compute_full_padding_for_dilated_conv(Tensor kernel_size, int dilation)
{
    // Convert IntList to Tensor to be able to use element-wise operations
    Tensor kernel_size_tensor = kernel_size.toType(CPU(kFloat));
                                  
    // Compute the actual kernel size after dilation
    auto actual_kernel_size = (kernel_size_tensor - 1) * (dilation - 1) + kernel_size_tensor;

    // Compute the padding size in order to achieve the 'full padding' mode
    auto full_padding = (actual_kernel_size / 2).floor_()
                                                .toType(CPU(kInt));
                                                  
    return full_padding;
};

torch::Module::Ptr torch::conv3x3(int in_planes, int out_planes, int stride, int dilation)
{
    // {3, 3} tuple in tensor form.
    // We need this because next function accepts Tensor
    Tensor kernel_size = CPU(kInt).tensor({2})
                                .fill_(3);

    Tensor padding = compute_full_padding_for_dilated_conv(kernel_size, dilation);

    auto padding_accessor = padding.accessor<int,1>(); 

    return std::make_shared<Conv2d>(in_planes,
                                    out_planes,
                                    3, 3,
                                    stride, stride,
                                    padding_accessor[0], padding_accessor[1],
                                    dilation, dilation,
                                    1, false);
};

torch::Module::Ptr torch::resnet_base_conv7x7()
{
    return make_shared<Conv2d>(
		3,      /* in_planes */
        64,     /* out_planes */
        7,      /* kernel_w */
        7,      /* kernel_h */
        2,      /* stride_w */
        2,      /* stride_h */
        3,      /* padding_w */
        3,      /* padding_h */
        1,      /* dilation_w */
        1,      /* dilation_h */
        1,      /* groups */
        false); /* bias */
}

torch::Module::Ptr torch::resnet_conv1x1(int in_planes, int planes)
{
    return std::make_shared<Conv2d>(in_planes, planes,
                                    1, 1,
                                    1, 1,
                                    0, 0,
                                    1, 1,
                                    1, false);
}    

Tensor torch::preprocess_batch(Tensor input_batch)
{
    // Subtracts mean and divides by std.
    // Important: image should be in a 0-1 range and not in 0-255

    // TODO: create a pull request to add broadcastable
    // operations

    auto mean_value = CPU(kFloat).ones({1, 3, 1, 1});

    mean_value[0][0][0][0] = 0.485f;
    mean_value[0][1][0][0] = 0.456f;
    mean_value[0][2][0][0] = 0.406f;

    // Broadcast the value
    auto mean_value_broadcasted = mean_value.expand(input_batch.sizes());

    auto std_value = CPU(kFloat).ones({1, 3, 1, 1});

    std_value[0][0][0][0] = 0.229f;
    std_value[0][1][0][0] = 0.224f;
    std_value[0][2][0][0] = 0.225f;

    auto std_value_broadcasted = std_value.expand(input_batch.sizes());

    return (input_batch - mean_value_broadcasted) / std_value_broadcasted;
}

void torch::inspect_checkpoint(string hdf5_filename)
{
    auto dict = load(hdf5_filename);

    for (auto name_tensor_pair : dict)
    {
    cout << name_tensor_pair.first << ": " << name_tensor_pair.second.sizes() <<endl;
    }
}

Tensor torch::upsample_bilinear(Tensor input_tensor, int output_height, int output_width)
{
    Tensor output = input_tensor.type().tensor();

    SpatialUpSamplingBilinear_updateOutput(input_tensor, output, output_height, output_width);

    return output;
}

Tensor torch::downsample_average(Tensor input_tensor, int downsample_factor)
{
    // Temporary solution for downsampling:
    // https://discuss.pytorch.org/t/how-to-simpler-downsample-an-image-tensor-with-bicubic/1296/2

    // Helper functions
    // http://nbviewer.jupyter.org/gist/tnarihi/54744612d35776f53278
    auto get_kernel_size = [](int factor) 
    { 

    return 2 * factor - factor % 2;
    };

    auto get_pading_size = [](int factor) 
    { 

    return int(ceil((factor - 1) / 2.));
    };

    Tensor output = input_tensor.type().tensor();

    int kernel_size = get_kernel_size(downsample_factor);
    int padding = get_pading_size(downsample_factor);

    cout << "kernel size: " << kernel_size << ", padding: " << padding << endl; 


    SpatialAveragePooling_updateOutput(input_tensor,
                                        output,
                                        kernel_size,
                                        kernel_size,
                                        downsample_factor,
                                        downsample_factor,
                                        padding,
                                        padding,
                                        false,
                                        true);

    return output;
}

Tensor torch::softmax(Tensor input_tensor)
{
    Tensor output = input_tensor.type().tensor();

    SoftMax_updateOutput(input_tensor, output);

    return output;
}


// TODO: explicit usage of Opencv's mat -- maybe try to hide it
// so that opencv won't be necessary for the main code

// Just put it in the opencv example file instead of the main library
Tensor torch::convert_opencv_mat_image_to_tensor(Mat input_mat)
{
    // Returns Byte Tensor with 0-255 values and (height x width x 3) shape
    // TODO: 
    // (1) double-check if this kind of conversion will always work
    //     http://docs.opencv.org/3.1.0/d3/d63/classcv_1_1Mat.html in 'Detailed Description'
    // (2) so far only works with byte representation of Mat

    unsigned char *data = (unsigned char*)(input_mat.data);

    int output_height = input_mat.rows;
    int output_width = input_mat.cols;

    auto output_tensor = CPU(kByte).tensorFromBlob(data, {output_height, output_width, 3});

    return output_tensor;
}

Tensor torch::convert_image_to_batch(Tensor input_img)
{
    // Converts height x width x depth Tensor to
    // 1 x depth x height x width Float Tensor

    // It's necessary because network accepts only batches

    auto output_tensor =  input_img.transpose(0, 2)
                                    .transpose(1, 2)
                                    .unsqueeze(0);

    return output_tensor;
}