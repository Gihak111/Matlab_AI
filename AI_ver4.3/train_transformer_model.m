function train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = enc.NumWords;
    numHeads = 4;
    maxLength = size(X{1}, 1);
    
    % Word Embedding Layer 설정
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords, 'wordEmbedding');
    
    % Transformer Layer 설정
    transformerLayerObj = TransformerLayer(numHeads, embeddingDimension, 1, 256, 0.1, 'transformerLayer');
    
    % Global Average Pooling Layer 설정
    globalAveragePoolingLayerObj = globalAveragePooling1dLayer();  % 이름 지정 없이 기본 생성자 사용
    
    % Fully Connected Layer 설정
    fullyConnectedLayerObj = fullyConnectedLayer(32, 'Name', 'fc');
    
    % Softmax Layer 설정
    softmaxLayerObj = softmaxLayer('Name', 'softmax');
    
    % Classification Output Layer 설정
    classificationLayerObj = classificationLayer('Name', 'classification');
    
    % Layer 배열
    layers = [
        sequenceInputLayer([maxLength 1], 'Name', 'input')
        wordEmbeddingLayerObj
        transformerLayerObj
        globalAveragePoolingLayerObj
        fullyConnectedLayerObj
        softmaxLayerObj
        classificationLayerObj
    ];

    % 학습 옵션 설정
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'Plots', 'training-progress');
    
    % 모델 훈련
    model = trainNetwork(X, Y, layers, options);

    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc');
end
