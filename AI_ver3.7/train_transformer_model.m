function train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = enc.NumWords;
    numHeads = 4;
    dim = embeddingDimension;
    depth = 1;
    feedForwardSize = 256;
    dropout = 0.1;
    numEpochs = 5;
    miniBatchSize = 32;

    % 모델 생성
    model = CustomTransformerModel(embeddingDimension, numWords, numHeads, dim, depth, feedForwardSize, dropout);

    % 학습 옵션 설정
    options = trainingOptions('adam', ...
        'MaxEpochs', numEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    % 데이터 전처리
    numMiniBatches = ceil(size(X, 1) / miniBatchSize);
    
    for epoch = 1:numEpochs
        disp(['Epoch ', num2str(epoch), ' of ', num2str(numEpochs)]);
        for miniBatch = 1:numMiniBatches
            % 미니배치 데이터 준비
            idx = (miniBatch-1) * miniBatchSize + 1:min(miniBatch * miniBatchSize, size(X, 1));
            miniBatchX = X(idx);
            miniBatchY = Y(idx);
            
            % 예측 및 손실 계산
            [loss, gradients] = model.modelGradients(miniBatchX, miniBatchY);
            
            % 모델 파라미터 업데이트
            [model.LearnableParameters, model.Velocity] = adamupdate(model.LearnableParameters, gradients, model.Velocity, options.LearnRate, options.GradientDecayFactor, options.SquaredGradientDecayFactor, options.Epsilon);
            
            % 손실 출력
            disp(['Mini-batch ', num2str(miniBatch), ' of ', num2str(numMiniBatches), ': Loss = ', num2str(extractdata(loss))]);
        end
    end
    
    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc');
end
