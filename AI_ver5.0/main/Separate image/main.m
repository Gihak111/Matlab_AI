% 1. 데이터 불러오기
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', 'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% 데이터셋 크기
numImages = numel(digitData.Files);

% 2. 신경망 구성
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(5,20,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,50,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% 3. 옵션 설정 및 신경망 학습
options = trainingOptions('sgdm', ...
    'MaxEpochs',50, ... % 최대 epoch 수
    'MiniBatchSize',numImages, ... % Iteration per epoch = 1
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(digitData,layers,options);

% 4. 테스트 데이터 평가
digitTestPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
digitTest = imageDatastore(digitTestPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
YPred = classify(net,digitTest);
YTest = digitTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest);
fprintf('정확도: %.2f%%\n', accuracy*100);
