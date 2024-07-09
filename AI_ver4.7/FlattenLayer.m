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
            % Flatten 처리: 입력 X의 마지막 두 차원을 하나의 차원으로 합침 => X를 [특징 수 x 1] 형태로 변환
             %Z = reshape(X, size(X,1), []);%첫 번째 차원을 유지하고 나머지를 하나로 합침 실패
            %Z = reshape(X, [], size(X,3)); %유일하게 특징 수 곱하기 1의 형태를 구현하는데 성공.
            %하지만 오류남 

            Z = reshape(X, [], 1);  % X를 [특징 수 x 1] 형태로 변환
            disp(size(Z))
            %ClassificationOutputLayer는 하나의 차원만 가져야 한다.
        end
    end
end
