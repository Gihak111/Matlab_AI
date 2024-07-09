function train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = enc.NumWords;
    numHeads = 4;
    maxLength = size(X{1}, 1); % Assuming all sequences have the same length
    
    % Word Embedding Layer 설정
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords);

    % Transformer Layer 설정
    layers = [
        sequenceInputLayer([maxLength 1]) % 입력 크기를 맞춤
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        TransformerLayer(numHeads, embeddingDimension, 1, 256, 0.1, wordEmbeddingLayerObj, 'transformerLayer') % Transformer 레이어
        globalAveragePooling1dLayer % 글로벌 평균 풀링 레이어
        flattenLayer % Flatten 레이어 추가
        fullyConnectedLayer(2) % 완전 연결 레이어
        softmaxLayer % 소프트맥스 레이어
        classificationLayer % 분류 레이어
    ];

    % 훈련 옵션 설정
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 32, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % 각 레이어의 정보 출력
    for i = 1:numel(layers)
        disp(['Layer ', num2str(i), ': ', class(layers(i))]);
        if isa(layers(i), 'nnet.cnn.layer.SequenceInputLayer')
            disp(['Input Size: ', mat2str(layers(i).InputSize)]);
        elseif isa(layers(i), 'wordEmbeddingLayer')
            disp(['Output Size: ', mat2str(wordEmbeddingLayerObj.getOutputSize())]);
        elseif isa(layers(i), 'TransformerLayer')
            disp('TransformerLayer: Input size depends on the previous layer (WordEmbeddingLayer).');
        else
            disp('Layer details: ');
            disp(layers(i));
        end
    end

    % 데이터 포맷 변환 없이 직접 사용
    % 모델 훈련
    model = trainNetwork(X, Y, layers, options);

    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc');
end
