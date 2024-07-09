classdef FlattenLayer < nnet.layer.Layer
    methods
        function layer = FlattenLayer(name)
            % 생성자: 이름 설정
            if nargin > 0
                layer.Name = name;
            end
        end
        
        function Z = predict(~, X)
            % Flatten 처리: 입력 X의 마지막 두 차원을 하나의 차원으로 합침
            Z = reshape(X, [], 1); % 모든 차원을 하나의 차원으로 합침
        end
    end
end
