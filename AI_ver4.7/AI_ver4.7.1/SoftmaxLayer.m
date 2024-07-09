classdef SoftmaxLayer < nnet.layer.Layer
    methods
        function Z = predict(~, X)
            Z = softmax(X);
        end
    end
end
