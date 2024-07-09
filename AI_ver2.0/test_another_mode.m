function combined_online_learning_supervised()
    % Initialize empty model and encoder
    model = [];
    enc = [];
    
    % Set initial training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 1, ... % 한 번에 하나의 데이터 배치만큼만 훈련
        'MiniBatchSize', 1, ... % 배치 크기 1로 설정하여 온라인 학습
        'Verbose', true, ...
        'Plots', 'training-progress');
    disp('options config');
    
    % Loop for online learning
    for iter = 1:10 % 예시로 10번의 반복
        disp(['Iteration: ' num2str(iter)]);

        disp('start get data from web');
        
        % 데이터 가져오기
        hn_data = hackernews_crawler();
        disp(hn_data);
        
        reddit_data = reddit_crawler();
        disp(reddit_data);
        
        % 데이터 전처리 및 클러스터링 (딥러닝 기반)
        try
            % Combine news and reddit data
            all_data = [hn_data; reddit_data];

            % Extract texts as a cell array of strings
            texts = {all_data.title}; % 각 title을 cell 배열에서 문자열로 추출

            % Convert cell array of strings to string array
            texts = string(texts); % cell 배열을 문자열 배열로 변환

            % TF-IDF 모델 학습
            if isempty(enc)
                enc = fit_tfidf(texts);
            end

            % Ensure proper length for hn_data and reddit_data
            num_hn = length(hn_data);
            num_reddit = length(reddit_data);

            % Create features (TF-IDF 예시)
            features = encode_tfidf(enc, texts);

            % Assign labels or cluster assignments (예시로 랜덤 라벨링)
            labels_hn = randi([1 3], num_hn, 1); % 예시로 3개의 클러스터

            % Prepare clustered data structures
            clustered_hn = struct('X', features(1:num_hn, :), 'Y', labels_hn);
            disp(clustered_hn);

            if num_reddit > 0
                clustered_reddit = struct('X', features(num_hn+1:end, :), 'Y', []);
            else
                clustered_reddit = [];
            end
            disp(clustered_reddit);

        catch ME
            disp(['Error in clustering: ' ME.message]);
            clustered_hn = [];
            clustered_reddit = [];
        end
        
        % 결과 요약
        summarize_results(clustered_hn, clustered_reddit);
        
        % 훈련된 모델 또는 클러스터 업데이트
        if isempty(model)
            % 최초 모델 생성 및 훈련 (예시로 RNN 사용)
            model = train_supervised_model(clustered_hn, 3); % 예시로 3개의 클러스터 사용
        else
            % 기존 모델 업데이트
            model = update_supervised_model(model, clustered_hn);
        end
        
        % 결과 비교 실행
        compare_opinions(model, clustered_hn, clustered_reddit);
    end
end