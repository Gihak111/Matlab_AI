% 1. 데이터 불러오기
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', 'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% 데이터 증강 설정
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-3,3], ...
    'RandYTranslation',[-3,3]);

augDigitData = augmentedImageDatastore([28 28 1], digitData, 'DataAugmentation', imageAugmenter);

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

% 3. 옵션 설정
options = trainingOptions('sgdm', ...
    'MaxEpochs',40, ... % 최대 에포크 수 증가
    'Verbose',false, ...
    'Plots','training-progress');

% 4. 학습 및 평가 반복
numRepeats = 10;
accuracies = zeros(1, numRepeats);

for i = 1:numRepeats
    % 신경망 학습
    net = trainNetwork(augDigitData, layers, options);
    
    % 테스트 데이터 평가
    digitTestPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
    digitTest = imageDatastore(digitTestPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    YPred = classify(net, digitTest);
    YTest = digitTest.Labels;

    accuracy = sum(YPred == YTest) / numel(YTest);
    accuracies(i) = accuracy;
    fprintf('반복 %d - 정확도: %.2f%%\n', i, accuracy*100);
end

% 평균 정확도 출력
meanAccuracy = mean(accuracies);
fprintf('평균 정확도: %.2f%%\n', meanAccuracy*100);
