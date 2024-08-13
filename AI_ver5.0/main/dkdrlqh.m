

% MATLAB 코드

% TensorFlow 모델 불러오기
savedModel = importTensorFlowNetwork('C:\Users\연준모\Documents\MATLAB\7월 2일\saprate_AI\turning_point\AI_ver5.0\main');

% 데이터 불러오기 (예시)
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
digitData = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 옵션 설정 및 학습
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Verbose', true);

net = trainNetwork(digitData, savedModel.Layers, options);

% 테스트 데이터 평가
YPred = classify(net, digitTest);
YTest = digitTest.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('정확도: %.2f%%\n', accuracy * 100);
