classdef GlobalAveragePooling1DLayer < nnet.layer.Layer

    properties
        % Define properties here, if any
    end

    methods
        function layer = GlobalAveragePooling1DLayer(name)
            disp("start function layer name")
            % Create a GlobalAveragePooling1DLayer with the name provided
            layer.Name = name;
            % Define any other properties or settings here
        end

        function Z = predict(layer, X)
            disp("입력 전 값 layer")
            disp(size(layer))
            disp("입력 전 값 X")
            disp(size(X))
            % Implement the forward propagation of the layer
            % X is the input to the layer
            % Z is the output of the layer
            Z = mean(X, 2); % Compute the mean along the sequence length dimension
            Z = squeeze(Z); % Remove singleton dimensions if any
            disp("출력 값")
            disp(size(Z))
            save z
        end
    end
end
