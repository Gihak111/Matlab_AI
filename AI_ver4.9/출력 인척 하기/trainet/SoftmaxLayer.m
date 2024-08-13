classdef SoftmaxLayer < nnet.layer.Layer
    methods
        function layer = SoftmaxLayer(name)
            layer.Name = name;
        end
        
        function Z = predict(obj, X)
            % Convert X to dlarray with 'SSCB' data format
            %dlX = dlarray(X, 'SSCB');
            
            % Apply softmax along the third dimension
            Z = softmax(X, 'DataFormat', 'SSCB');
            
        end


    end
end
