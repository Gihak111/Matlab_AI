classdef FullyConnectedLayer_self < nnet.layer.Layer
    properties
        Weights
        Bias
        OutputSize
    end
    
    methods
        function layer = FullyConnectedLayer_self(numOutputs, name)
            % 생성자: 이름과 출력 크기 설정
            layer.Name = name;
            layer.OutputSize = numOutputs;
            layer.Weights = [];
            layer.Bias = randn([numOutputs, 1]);
        end
        
        function Z = predict(layer, X)
            % 입력 데이터에 가중치 행렬을 곱하고 편향 벡터를 더함
            numInputs = size(X, 2);
            if isempty(layer.Weights)
                layer.Weights = randn([layer.OutputSize, numInputs]);
            end
            Z = X * layer.Weights' + layer.Bias';
            disp("완전연결계층 통과")
            disp(size(Z))
        end
    end
end
