function model = train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = enc.NumWords;
    numHeads = 4;
    numTransformers = 1; % 단일 Transformer 레이어 사용
    intermediateDim = 256;
    dropoutRate = 0.1;
    
    % Word Embedding Layer 초기화
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords);
    
    % Transformer Layer 초기화
    transformerLayer = TransformerLayer(numHeads, embeddingDimension, numTransformers, intermediateDim, dropoutRate, wordEmbeddingLayerObj, 'transformerLayer');
    
    % 레이어 배열 정의
    maxLength = size(X{1}, 1); % 모든 시퀀스가 같은 길이일 것으로 가정
    layers = [
        sequenceInputLayer([maxLength 1]) % 입력 크기 설정
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        transformerLayer % Transformer 레이어
        softmaxLayer('Name', 'softmax') % Softmax 레이어
        classificationLayer('Name', 'classification') % 분류 레이어
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