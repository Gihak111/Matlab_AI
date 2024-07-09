classdef SoftmaxLayer < nnet.layer.Layer
    methods
        function layer = SoftmaxLayer(name)
            % 생성자: 이름 설정
            if nargin > 0
                layer.Name = name;
            end
        end
        
        function Z = predict(obj, X)
            % Softmax 처리: 입력 X의 마지막 차원에 softmax 적용
            Z = softmax(X, 'DataFormat', 'SSCB');
        end
    end
end
