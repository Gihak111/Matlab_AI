classdef FullyConnectedLayer < nnet.layer.Layer
    properties
        InputSize
        OutputSize
    end
    
    properties (Learnable)
        Weights
        Bias
    end
    
    methods
        function layer = FullyConnectedLayer(inputSize, outputSize, name)
            layer.InputSize = inputSize;
            layer.OutputSize = outputSize;
            layer.Name = name;
            
            % Initialize learnable parameters
            layer.Weights = dlarray(randn(outputSize, inputSize, 'single'));
            layer.Bias = dlarray(randn(outputSize, 1, 'single'));
        end
        
        function Z = predict(layer, X)
            Z = layer.Weights * X + layer.Bias;
        end
    end
end
