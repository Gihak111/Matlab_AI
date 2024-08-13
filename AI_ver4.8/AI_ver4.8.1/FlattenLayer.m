%{
classdef FlattenLayer < nnet.layer.Layer
    methods
        function layer = FlattenLayer(name)
            % 생성자: 이름 설정
            if nargin > 0
                layer.Name = name;
            end
        end
        
        function Z = predict(~, X)
            disp(size(X))
            % 입력의 첫 번째 차원은 배치 크기, 나머지를 모두 하나로 합침
            Z = reshape(X, size(X,1), []); 
            % 이제 2차원 배열을 4차원 배열로 변환
            Z = reshape(Z, size(X,1), 1, 1, []);
            disp(size(Z))
        end
    end
end
%}

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
