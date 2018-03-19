#include "pytorch.h"
#include "ResNet.hxx"

torch::Resnet18_8s::Resnet18_8s(int num_classes):
            num_classes(num_classes)
{
	resnet18_8s = torch::resnet18(num_classes,    
									true,           /* fully convolutional model */
									8,              /* we want subsampled by 8 prediction*/
									true);          /* remove average pooling layer */

	// Adding a module with this name to be able to easily load
	// weights from pytorch models
	add_module("resnet18_8s", resnet18_8s);
}

torch::Resnet18_8s::~Resnet18_8s()
{

}

Tensor torch::Resnet18_8s::forward(Tensor input)
{
	// probably we can add some utility functions to add softmax on top 
	// resize the ouput in a proper way

	// input is a tensor of shape batch_size x #channels x height x width
	int output_height = input.size(2);
	int output_width = input.size(3);

	auto subsampled_prediction = resnet18_8s->forward(input);

	auto full_prediction = at::upsample_bilinear2d(subsampled_prediction, {output_height, output_width});

	return full_prediction;
}
     
torch::Resnet34_8s::Resnet34_8s(int num_classes):
            num_classes(num_classes)
{
	resnet34_8s = torch::resnet34(num_classes,    
									true,           /* fully convolutional model */
									8,              /* we want subsampled by 8 prediction*/
									true);          /* remove average pooling layer */

	// Adding a module with this name to be able to easily load
	// weights from pytorch models
	add_module("resnet34_8s", resnet34_8s);
}

torch::Resnet34_8s::~Resnet34_8s()
{

}

Tensor torch::Resnet34_8s::forward(Tensor input)
{
	// TODO:

	// (1) This part with upsampling is the same for all fully conv models
	//     Might make sense to write an abstract class to avoid duplication
	// (2) Probably we can add some utility functions to add softmax on top 
	//      resize the ouput in a proper way

	// input is a tensor of shape batch_size x #channels x height x width
	int output_height = input.size(2);
	int output_width = input.size(3);

	auto subsampled_prediction = resnet34_8s->forward(input);

	auto full_prediction = at::upsample_bilinear2d(subsampled_prediction, {output_height, output_width});

	return full_prediction;
}


 torch::Module::Ptr torch::resnet18(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
 {
   return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
          new torch::ResNet<torch::BasicBlock>({2, 2, 2, 2},
                                               num_classes,
                                               fully_conv,
                                               remove_avg_pool,
                                               output_stride ));
 }


 torch::Module::Ptr torch::resnet34(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
 {
   return std::shared_ptr<torch::ResNet<torch::BasicBlock>>(
          new torch::ResNet<torch::BasicBlock>({3, 4, 6, 3},
                                               num_classes,
                                               fully_conv,
                                               remove_avg_pool,
                                               output_stride ));
 }

 torch::Module::Ptr torch::resnet50(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
 {
   return std::shared_ptr<torch::ResNet<torch::Bottleneck>>(
          new torch::ResNet<torch::Bottleneck>({3, 4, 6, 3},
                                               num_classes,
                                               fully_conv,
                                               remove_avg_pool,
                                               output_stride ));
 }

 torch::Module::Ptr torch::resnet101(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
 {
   return std::shared_ptr<torch::ResNet<torch::Bottleneck>>(
          new torch::ResNet<torch::Bottleneck>({3, 4, 23, 3},
                                               num_classes,
                                               fully_conv,
                                               remove_avg_pool,
                                               output_stride ));
 }

 torch::Module::Ptr torch::resnet152(int num_classes=1000, bool fully_conv=false, int output_stride=32, bool remove_avg_pool=false)
 {
   return std::shared_ptr<torch::ResNet<torch::Bottleneck>>(
          new torch::ResNet<torch::Bottleneck>({3, 8, 36, 3},
                                               num_classes,
                                               fully_conv,
                                               remove_avg_pool,
                                               output_stride ));
 }


 // Maybe add new options like add_softmax?,
 torch::Module::Ptr torch::resnet18_imagenet()
 {
   return resnet18(1000, false, 32, false);
 }

 torch::Module::Ptr torch::resnet34_imagenet()
 {
   return resnet34(1000, false, 32, false);
 }

 torch::Module::Ptr torch::resnet50_imagenet()
 {
   return resnet50(1000, false, 32, false);
 }

 torch::Module::Ptr torch::resnet101_imagenet()
 {
   return resnet101(1000, false, 32, false);
 }

 torch::Module::Ptr torch::resnet152_imagenet()
 {
   return resnet152(1000, false, 32, false);
 }

 torch::Module::Ptr torch::resnet18_8s_pascal_voc()
 {

   return make_shared<torch::Resnet18_8s>(21);
 }

 torch::Module::Ptr torch::resnet34_8s_pascal_voc()
 {
   return make_shared<torch::Resnet34_8s>(21);
 }
