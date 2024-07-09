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
    
    % Loop for online learning
    for iter = 1:10 % 예시로 10번의 반복
        disp(['Iteration: ' num2str(iter)]);
        
        % 데이터 가져오기
        hn_data = hackernews_crawler();
        reddit_data = reddit_crawler();
        
        % Combine news and reddit data
        all_data = [hn_data; reddit_data];
        
        % Extract texts as a cell array of strings
        texts = {all_data.title}; 
        
        % Convert cell array of strings to string array
        texts = string(texts); 
        
        try
            % Create TF-IDF model if not initialized
            if isempty(enc)
                enc = fit_tfidf(texts);
            end
            
            % Encode texts using TF-IDF
            features = encode_tfidf(enc, texts);
            
            % Prepare clustered data structures
            num_hn = length(hn_data);
            num_reddit = length(reddit_data);
            
            labels_hn = randi([1 3], num_hn, 1); % 예시로 3개의 클러스터
            
            % Prepare clustered data
            clustered_hn = struct('X', features(1:num_hn, :), 'Y', labels_hn);
            clustered_reddit = struct('X', features(num_hn+1:end, :), 'Y', []);
            
            % 결과 요약
            summarize_results(clustered_hn, clustered_reddit, texts, enc);
            
            % 훈련된 모델 또는 클러스터 업데이트
            if isempty(model)
                model = train_supervised_model(clustered_hn, 3); % 예시로 3개의 클러스터 사용
            else
                model = update_supervised_model(model, clustered_hn);
            end
            
            % 결과 비교 실행
            compare_opinions(model, clustered_hn, clustered_reddit);
            
        catch ME
            disp(['Error during iteration ' num2str(iter) ': ' ME.message]);
        end
    end
end