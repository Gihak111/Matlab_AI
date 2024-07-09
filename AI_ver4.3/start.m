function main()
    % 데이터 로드
    hn_data = hackernews_crawler();
    reddit_data = reddit_crawler();

    % 데이터 전처리
    [X, Y, enc] = preprocess_data(hn_data, reddit_data);

    % 모델 훈련
    train_transformer_model(X, Y, enc);

    % 예측 및 결과 비교 함수
    compare_opinions(hn_data, reddit_data);
end