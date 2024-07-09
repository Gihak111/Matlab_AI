%{
function train_transformer_model(X, Y, enc)
    numWords = enc.NumWords; % 단어 사전의 크기
    maxLength = size(X, 1); % 최대 시퀀스 길이

    % Word Embedding 설정
    embeddingDimension = 128;
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords);
    wordEmbeddingOutputSize = wordEmbeddingLayerObj.getOutputSize();

    layers = [
        sequenceInputLayer([maxLength 1]) % 입력 크기 설정
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        TransformerLayer(4, embeddingDimension, 1, 256, 0.1, wordEmbeddingLayerObj, 'transformerLayer') % Transformer 레이어
        globalAveragePooling1dLayer % 글로벌 평균 풀링 레이어
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
            disp(['Output Size: ', mat2str(wordEmbeddingOutputSize)]);
        elseif isa(layers(i), 'TransformerLayer')
            disp('TransformerLayer: Input size depends on the previous layer (WordEmbeddingLayer).');
        else
            disp('Layer details: ');
            disp(layers(i));
        end
    end

    % 모델 훈련
    model = trainNetwork(X, Y, layers, options); % 모델 훈련

    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc'); % 모델과 인코더 저장
end
%}


function train_transformer_model(X, Y, enc)
    numWords = enc.NumWords; % 단어 사전의 크기
    maxLength = size(X, 1); % 최대 시퀀스 길이

    % Word Embedding 설정
    embeddingDimension = 128;
    wordEmbeddingLayerObj = wordEmbeddingLayer(embeddingDimension, numWords);
    
    layers = [
        sequenceInputLayer([maxLength 1]) % 입력 크기 설정
        wordEmbeddingLayerObj % 단어 임베딩 레이어
        TransformerLayer(4, embeddingDimension, 1, 256, 0.1, wordEmbeddingLayerObj, 'transformerLayer') % Transformer 레이어
        globalAveragePooling1dLayer % 글로벌 평균 풀링 레이어
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

    % 모델 훈련
    model = trainNetwork(X, Y, layers, options); % 모델 훈련

    % 모델과 인코더 저장
    save('news_model_transformer.mat', 'model', 'enc'); % 모델과 인코더 저장
end
