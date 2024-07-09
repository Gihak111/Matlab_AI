classdef SequenceInputLayer < nnet.layer.Layer
    properties
        InputSize
    end
    
    methods
        function layer = SequenceInputLayer(inputSize, name)
            layer.InputSize = inputSize;
            layer.Name = name;
        end
        
        function Z = predict(layer, X)
            Z = X;
        end
    end
end
