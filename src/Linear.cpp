#include "pytorch.h"

torch::Linear::Linear( int in_features,
        int out_features,
        bool bias) :
        in_features(in_features),
        out_features(out_features),
        bias(bias)
{
    // Initialize weights here

    parameters["weight"] = TENSOR_DEFAULT_TYPE.zeros({out_features, in_features});

    // Check if we need bias for our convolution
    if(bias)
    {

    parameters["bias"] = TENSOR_DEFAULT_TYPE.ones({out_features});
    }
    else
    {

    // don't know why this works yet, doesn't work with TENSOR_DEFAULT_TYPE.tensor();
    parameters["bias"] = Tensor();
    }
};

torch::Linear::~Linear() 
{

};

string torch::Linear::tostring(int indentation_level)
{

    std::stringstream string_stream;

    string indentation = string(indentation_level, ' ');

    string_stream << indentation
                << "nn.Linear( "
                << "in_features=" << std::to_string(in_features) << " "
                << "out_features=" << std::to_string(out_features) << " "
                << "bias=" << std::to_string(bias) << " )";

    return string_stream.str();

};

Tensor torch::Linear::forward(Tensor input)
{
    // https://github.com/pytorch/pytorch/blob/49ec984c406e67107aae2891d24c8839b7dc7c33/torch/nn/_functions/linear.py

    Tensor output = input.type().zeros({input.size(0), parameters["weight"].size(0)});

    output.addmm_(0, 1, input, parameters["weight"].t());
         
    if(bias)
    {
    // TODO: check if in-place resize affects the result
    output.add_(parameters["bias"].expand({output.size(0), output.size(1)}));  
    }
         
    return output; 
};
