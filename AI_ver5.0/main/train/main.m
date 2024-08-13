% Transformer 모델을 위한 단순화된 예시 코드

% Hyperparameters
sequenceLength = 10; % 시퀀스 길이
embeddingDimension = 128; % 임베딩 차원
numHeads = 2; % Self-Attention 헤드 수
hiddenDimension = 256; % Feedforward 네트워크 은닉층 차원

% 입력 레이어
inputLayer = sequenceInputLayer(sequenceLength, 'Name', 'input');

% 단어 임베딩 레이어 (내장된 wordEmbeddingLayer 사용)
wordEmbeddingLayer = wordEmbeddingLayer(embeddingDimension, 'Name', 'wordEmbedding');

% Transformer 레이어
selfAttentionLayer = multiHeadAttentionLayer(numHeads, embeddingDimension, 'SelfAttention');
feedforwardLayer = [
    fullyConnectedLayer(hiddenDimension, 'Name', 'fc1');
    reluLayer('Name', 'relu');
    fullyConnectedLayer(embeddingDimension, 'Name', 'fc2');
];

% 레이어 그래프 구성
lgraph = layerGraph();
lgraph = addLayers(lgraph, inputLayer);
lgraph = addLayers(lgraph, wordEmbeddingLayer);
lgraph = addLayers(lgraph, selfAttentionLayer);
lgraph = addLayers(lgraph, feedforwardLayer);

% 레이어 연결
lgraph = connectLayers(lgraph, 'input', 'wordEmbedding');
lgraph = connectLayers(lgraph, 'wordEmbedding', 'SelfAttention/in');
lgraph = connectLayers(lgraph, 'SelfAttention/out', 'fc1');

% 출력 레이어
outputLayer = [
    fullyConnectedLayer(2, 'Name', 'fc_final');
    softmaxLayer('Name', 'softmax');
    classificationLayer('Name', 'classification');
];
lgraph = addLayers(lgraph, outputLayer);

% 레이어 그래프 시각화
figure;
plot(lgraph);

% 데이터 생성 및 더미 레이블 생성
X = randn(1, sequenceLength, embeddingDimension); % 예시 데이터
Y = categorical(randi([1 2], 1, 1)); % 예시 레이블

% 훈련 옵션 설정
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress');

% 모델 훈련
net = trainNetwork(X, Y, lgraph, options);
