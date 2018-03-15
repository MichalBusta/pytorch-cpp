#ifndef _RESNET_HXX_
#define _RESNET_HXX_

// First depth input is the same for all resnet models
template <class BlockType>
torch::ResNet<BlockType>::ResNet(
	IntList layers,
	int num_classes,
	bool fully_conv,
	bool remove_avg_pool,
	int output_stride) :
	in_planes(64),
	output_stride(output_stride),
	fully_conv(fully_conv),
	remove_avg_pool(remove_avg_pool)
{
	// Stride is four after first convolution and maxpool layer.
	// We use this class member to track current output stride in make_layer()
	current_stride = 4;

	// Dilation hasn't been applied after convolution and maxpool layer.
	// We use this class member to track dilation factor in make_layer()
	current_dilation = 1;

	conv1 = resnet_base_conv7x7();
	bn1 = std::make_shared<BatchNorm2d>(64);
	relu = std::make_shared<ReLU>();
	// Kernel size: 3, Stride: 2, Padding, 1 -- full padding 
	maxpool = std::make_shared<MaxPool2d>(3, 3, 2, 2, 1, 1);

	layer1 = make_layer(64, layers[0], 1);
	layer2 = make_layer(128, layers[1], 2);
	layer3 = make_layer(256, layers[2], 2);
	layer4 = make_layer(512, layers[3], 2);

	avgpool = std::make_shared<AvgPool2d>(7, 7);
	fc = std::make_shared<Linear>(512 * BlockType::expansion, num_classes);

	if (fully_conv)
	{
		// Average pooling with 'full padding' mode
		avgpool = std::make_shared<AvgPool2d>(7, 7,
			1, 1,
			3, 3);

		// 1x1 Convolution -- Convolutionalized Linear Layer
		fc = std::make_shared<Conv2d>(512 * BlockType::expansion,
			num_classes,
			1, 1);
	}

	add_module("conv1", conv1);
	add_module("bn1", bn1);
	add_module("relu", relu);

	add_module("maxpool", maxpool);

	add_module("layer1", layer1);
	add_module("layer2", layer2);
	add_module("layer3", layer3);
	add_module("layer4", layer4);

	add_module("avgpool", avgpool);

	add_module("fc", fc);

	module_name = "ResNet";
}

template <class BlockType>
torch::ResNet<BlockType>::~ResNet()
{

}

template <class BlockType>
Tensor torch::ResNet<BlockType>::forward(Tensor input)
{
	Tensor output = input.type().tensor();

	output = conv1->forward(input);
	output = bn1->forward(output);
	output = relu->forward(output);
	output = maxpool->forward(output);

	output = layer1->forward(output);
	output = layer2->forward(output);
	output = layer3->forward(output);
	output = layer4->forward(output);

	if(!remove_avg_pool)
	{
	    output = avgpool->forward(output);
	}

	if(!fully_conv)
	{
	    // Flatten the output in order to apply linear layer
	    output = output.view({output.size(0), -1});
	}

	output = fc->forward(output);

	return output;
}

template <class BlockType>
torch::Module::Ptr torch::ResNet<BlockType>::make_layer(int planes, int blocks, int stride)
{
	auto new_layer = std::make_shared<torch::Sequential>();

	Module::Ptr downsample = nullptr;

	// Check if we need to downsample
	if(stride != 1 || in_planes != planes * BlockType::expansion)
	{

		// See if we already achieved desired output stride
		if(current_stride == output_stride)
		{

		// If so, replace subsampling with dilation to preserve
		// current spatial resolution
		current_dilation = current_dilation * stride;
		stride = 1;
		}
		else
		{

		// If not, we perform subsampling
		current_stride = current_stride * stride;
		}


		downsample = std::make_shared<torch::Sequential>();

		downsample->add( std::make_shared<torch::Conv2d>(in_planes,
														planes * BlockType::expansion,
														1, 1,
														stride, stride,
														0, 0,
														1, 1,
														1,
														false) );

		downsample->add(std::make_shared<BatchNorm2d>(planes * BlockType::expansion));

	}

	auto first_block = std::make_shared<BlockType>(in_planes,
													planes,
													stride,
													current_dilation,
													downsample);
	new_layer->add(first_block);

	in_planes = planes * BlockType::expansion;

	for (int i = 0; i < blocks - 1; ++i)
	{
         
		new_layer->add(std::make_shared<BlockType>(in_planes,
												planes,
												1,
												current_dilation));
	}

	return new_layer;
}

#endif
