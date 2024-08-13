classdef FlattenLayer < nnet.layer.Layer
    methods
        function obj = FlattenLayer(name)
            obj.Name = name;
        end
        
        function Z = predict(obj, X)
            % Flatten the input
            Z = my_reshape(X, size(X, 1), []);
        end
    end
end