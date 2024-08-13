classdef ClassificationLayer < nnet.layer.ClassificationLayer
    methods
        function layer = ClassificationLayer(name)
            % 생성자: 이름 설정
            layer.Name = name;
        end
        
        function loss = forwardLoss(layer, Y, T)
            disp("forwardLoss_start")
            % 소프트맥스 활성화 함수 적용
            Y = softmax(Y, 'DataFormat', 'SC');
            % 크로스 엔트로피 손실 계산
            loss = -sum(T .* log(Y), 'all');
            disp("forwardLoss_end")
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            disp("backwardLoss_start")
            % 소프트맥스 활성화 함수 적용
            Y = softmax(Y, 'DataFormat', 'SC');
            % 손실에 대한 그래디언트 계산
            dLdY = Y - T;
            disp("backwardLoss_end")
        end
    end
end
