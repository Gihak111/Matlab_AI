function train_transformer_model(X, Y, enc)
    % Hyperparameters
    embeddingDimension = 128;
    numWords = 2;%enc.NumWords;
    numHeads = 4;
    maxLength = size(X{1}, 1); % Assuming all sequences have the same length
    
    % Word Embedding Layer 설정
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords);

    % Transformer 인코더 레이어 설정
    transformerEncoderLayer = TransformerEncoderLayer(numHeads, embeddingDimension, 1, 256, 0.1, 'transformerEncoderLayer');

    % Transformer 디코더 레이어 설정
    transformerDecoderLayer = TransformerDecoderLayer(numHeads, embeddingDimension, 1, 256, 0.1, 'transformerDecoderLayer');

    % Softmax Layer 설정
    softmaxLayerObj = SoftmaxLayer('softmaxLayer');
    
    % Classification Output Layer 설정
    classificationLayerObj = ClassificationLayer('classificationLayer');
    
    % 레이어 배열 정의
    layers = [
        sequenceInputLayer([maxLength embeddingDimension]) % 입력 크기 설정
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        transformerEncoderLayer % Transformer 인코더 레이어
        transformerDecoderLayer % Transformer 디코더 레이어
        FlattenLayer % 시퀀스 데이터를 벡터 형태로 변환
        FullyConnectedLayer_self(numWords, 'fc1')
        softmaxLayerObj
        classificationLayerObj % 분류 레이어
    ];

    % 각 레이어의 정보 출력
    for i = 1:numel(layers)
        disp(['Layer ', num2str(i), ': ', class(layers(i))]);
        if isa(layers(i), 'nnet.cnn.layer.SequenceInputLayer')
            disp(['Input Size: ', mat2str(layers(i).InputSize)]);
        elseif isa(layers(i), 'wordEmbeddingLayer')
            disp(['Output Size: ', mat2str(wordEmbeddingLayerObj.getOutputSize())]);
        elseif isa(layers(i), 'TransformerEncoderLayer')
            disp(['TransformerEncoderLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'TransformerDecoderLayer')
            disp(['TransformerDecoderLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'GlobalAveragePooling1DLayer')
            disp(['GlobalAveragePooling1DLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
            disp(['FullyConnectedLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'SoftmaxLayer')
            disp(['SoftmaxLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'SqueezeLayer')
            disp(['SqueezeLayer: ', layers(i).Name]);
        elseif isa(layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            disp(['ClassificationOutputLayer: ', layers(i).Name]);
        else
            disp('Layer details: ');
            disp(layers(i));
        end
    end

    % 훈련 옵션 설정
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 64, ...
        'Plots', 'training-progress');

    % 모델 훈련
    model = trainNetwork(X, Y, layers, options);

    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc');
end