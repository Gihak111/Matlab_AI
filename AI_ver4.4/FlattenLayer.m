classdef FlattenLayer < nnet.layer.Layer
    methods
        function Z = predict(~, X)
            Z = X(:)';
        end
    end
end
