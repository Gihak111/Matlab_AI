function train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = enc.NumWords;
    numHeads = 4;
    maxLength = size(X{1}, 1); % Assuming all sequences have the same length
    
    % Word Embedding Layer 설정
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords, 'wordEmbedding');
    
    % Transformer Layer 설정
    transformerLayer = TransformerLayer(numHeads, embeddingDimension, 1, 256, 0.1, wordEmbeddingLayerObj, 'transformerLayer');
    
    % 레이어 배열 정의
    layers = [
        sequenceInputLayer([maxLength 1]) % 입력 크기 설정
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        transformerLayer % Transformer 레이어
    ];
    
    % 훈련 옵션 설정
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 32, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    % 모델 훈련
    model = trainNetwork(X, Y, layers, options);
    
    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc');
end