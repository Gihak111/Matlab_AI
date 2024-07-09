classdef ClassificationOutputLayer < nnet.layer.ClassificationLayer
    methods
        function Z = predict(~, X)
            Z = X;
        end
        
        function loss = forwardLoss(~, Y, T)
            % Calculate cross-entropy loss
            loss = crossentropy(Y, T);
        end
    end
end
