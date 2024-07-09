clear 
clc
%attentionWeights = softmax(attentionScores, 'DataFormat', 'SSB');
% 데이터 로드
hn_data = hackernews_crawler(); % Hacker News 데이터 크롤링
reddit_data = reddit_crawler(); % Reddit 데이터 크롤링

% 데이터 전처리
[X, Y, enc] = preprocess_data(hn_data, reddit_data); % 데이터 전처리

% 모델 훈련
train_transformer_model(X, Y, enc); % 모델 학습 및 저장

% 예측 및 결과 비교 함수
compare_opinions(); % 결과 비교 실행
