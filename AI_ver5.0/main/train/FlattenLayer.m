classdef FlattenLayer < nnet.layer.Layer
    methods
        function layer = FlattenLayer(name)
            % 생성자: 이름 설정
            if nargin > 0
                layer.Name = name;
            end
        end
        
        function Z = predict(~, X)
            % 입력의 첫 번째 차원은 배치 크기, 나머지를 모두 하나로 합침
            Z = reshape(X, size(X,1), []); 
            disp(size(Z))
        end
    end
end
