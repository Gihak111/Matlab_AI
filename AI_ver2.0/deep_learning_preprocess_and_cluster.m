function [clustered_hn, clustered_reddit, enc] = deep_learning_preprocess_and_cluster(hn_data, reddit_data, enc)
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
end
