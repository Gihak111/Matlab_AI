% 데이터 크롤러 함수 호출
reddit_data = reddit_crawler();
hn_data = hackernews_crawler();

% 데이터 병합 및 전처리
titles = {reddit_data.title, hn_data.title};
titles = string(titles);

% 텍스트 데이터 전처리
documents = tokenizedDocument(titles);
enc = wordEncoding(documents);

X = doc2sequence(enc, documents);
Y = categorical(ones(numel(X), 1)); % dummy labels, 필요에 따라 실제 레이블로 대체

% 각 문서의 최대 길이를 계산하여 패딩할 길이를 설정
maxSeqLength = max(cellfun(@(x) size(x, 2), X));
X = padsequences(X, 2, 'PaddingValue', 0, 'Length', maxSeqLength);

% 시퀀스 데이터를 특징 차원에 맞추기 위해 reshape
numFeatures = size(X{1}, 1);
for i = 1:numel(X)
    X{i} = reshape(X{i}, [numFeatures, 1, maxSeqLength]);
end

% LSTM 네트워크 아키텍처 정의
embeddingDimension = 50;
numHiddenUnits = 100;
numClasses = numel(unique(Y));

layers = [ ...
    sequenceInputLayer(numFeatures) % 입력 크기를 특징 차원으로 설정
    wordEmbeddingLayer(embeddingDimension, enc.NumWords)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% 옵션 설정
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'SequenceLength', 'shortest', ...
    'GradientThreshold', 2, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 모델 학습
net = trainNetwork(X, Y, layers, options);

% 모델 평가 (여기서는 간단히 훈련 데이터에 대한 평가)
YPred = classify(net, X);
accuracy = sum(YPred == Y) / numel(Y);
disp("Training Accuracy: " + accuracy);
