% SqueezeLayer 클래스 정의
classdef SqueezeLayer < nnet.layer.Layer
    methods
        function layer = SqueezeLayer(name)
            layer.Name = name;
        end

        function Z = predict(~, X)
            Z = squeeze(X);
        end
    end
end