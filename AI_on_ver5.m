function hn_data = hackernews_crawler()
    % URL 설정
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty';
    
    % 웹 옵션 설정
    options = weboptions('Timeout', 30);
    
    try
        % 최상위 스토리 ID 가져오기
        story_ids = webread(url, options);
        
        % 초기화
        hn_data = [];
        
        % 최대 10개의 스토리만 가져오기
        for i = 1:min(10, length(story_ids))
            try
                % 각 스토리의 URL 구성
                story_url = strcat('https://hacker-news.firebaseio.com/v0/item/', num2str(story_ids(i)), '.json?print=pretty');
                
                % 스토리 데이터 가져오기
                story = webread(story_url, options);
                
                % 제목과 링크 저장
                hn_data = [hn_data; struct('title', story.title, 'link', story.url)];
            catch ME
                disp(['Failed to fetch story ID: ' num2str(story_ids(i))]);
                disp(['Error Message: ' ME.message]);
            end
        end
    catch ME
        disp('Failed to fetch Hacker News data.');
        disp(['Error Message: ' ME.message]);
        hn_data = []; % 데이터를 가져오지 못한 경우 빈 배열 반환
    end
end
%%
function reddit_data = reddit_crawler()
    % URL 설정
    url = 'https://www.reddit.com/r/news/top/.json?limit=10';
    
    % 웹 옵션 설정
    options = weboptions('UserAgent', 'MATLAB', 'Timeout', 30);
    
    try
        % Reddit 데이터 가져오기
        data = webread(url, options);
        
        % 초기화
        reddit_data = [];
        
        % children에서 제목과 링크 추출
        children = data.data.children;
        for i = 1:length(children)
            post = children(i).data;
            reddit_data = [reddit_data; struct('title', post.title, 'link', post.url)];
        end
    catch ME
        disp('Failed to fetch Reddit data.');
        disp(['Error Message: ' ME.message]);
        reddit_data = []; % 데이터를 가져오지 못한 경우 빈 배열 반환
    end
end
%%
function summarize_results(clustered_hn, clustered_reddit)
    disp('--- Summarized Results ---');
    
    % Summarize Hacker News
    if ~isempty(clustered_hn)
        hn_titles = {clustered_hn.title};
        hn_links = {clustered_hn.link};
        disp('Hacker News Top Posts:');
        for i = 1:length(hn_titles)
            disp(['Title: ' hn_titles{i}]);
            disp(['Link: ' hn_links{i}]);
        end
    else
        disp('No Hacker News data available.');
    end
    
    % Summarize Reddit
    if ~isempty(clustered_reddit)
        reddit_titles = {clustered_reddit.title};
        reddit_links = {clustered_reddit.link};
        disp('Reddit Top Posts:');
        for i = 1:length(reddit_titles)
            disp(['Title: ' reddit_titles{i}]);
            disp(['Link: ' reddit_links{i}]);
        end
    else
        disp('No Reddit data available.');
    end
end
%%
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
        
        % 데이터 로드
        hn_data = hackernews_crawler();
        reddit_data = reddit_crawler();
        
        % 데이터 전처리 및 클러스터링 (딥러닝 기반)
        [clustered_hn, clustered_reddit, enc] = deep_learning_preprocess_and_cluster(hn_data, reddit_data, enc);
        
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
%%
function [clustered_hn, clustered_reddit, enc] = deep_learning_preprocess_and_cluster(hn_data, reddit_data, enc)
    try
        % Combine news and reddit data
        all_data = [hn_data; reddit_data];

        % Extract texts
        texts = {all_data.title}';

        % Create features (예시로 TF-IDF 사용)
        [features, enc] = create_tfidf_features(texts, enc);

        % Assign labels or cluster assignments (예시로 랜덤 라벨링)
        labels = randi([1 3], size(features, 1), 1); % 예시로 3개의 클러스터

        % Prepare clustered data structures
        clustered_hn = struct('title', {hn_data.title}, 'link', {hn_data.link}, 'labels', labels(1:length(hn_data)));
        clustered_reddit = struct('title', {reddit_data.title}, 'link', {reddit_data.link}, 'labels', labels(length(hn_data)+1:end));

    catch ME
        disp(['Error in clustering: ' ME.message]);
        clustered_hn = [];
        clustered_reddit = [];
    end
end
%%
function [features, enc] = create_tfidf_features(texts, enc)
    % Create TF-IDF features
    if isempty(enc)
        enc = fit_tfidf(texts);
    end
    features = encode_tfidf(enc, texts);
end
%%
function model = fit_tfidf(texts)
    % Fit TF-IDF model
    % Create vocabulary
    vocab = unique([texts{:}]);
    numDocs = length(texts);
    numWords = length(vocab);

    % Initialize term frequency (TF) and inverse document frequency (IDF) matrices
    tf = zeros(numDocs, numWords);
    df = zeros(1, numWords);

    for i = 1:numDocs
        words = texts{i};
        for j = 1:numWords
            tf(i, j) = sum(strcmp(words, vocab{j}));
        end
        df = df + (tf(i, :) > 0);
    end

    idf = log(numDocs ./ (df + 1));

    % Store TF-IDF model
    model.vocab = vocab;
    model.idf = idf;
end
%%
function features = encode_tfidf(model, texts)
    % Encode texts into TF-IDF vectors using the model
    numDocs = length(texts);
    numWords = length(model.vocab);
    features = zeros(numDocs, numWords);

    for i = 1:numDocs
        words = texts{i};
        for j = 1:numWords
            tf = sum(strcmp(words, model.vocab{j}));
            features(i, j) = tf * model.idf(j);
        end
    end
end
%%
function model = train_supervised_model(data, numClasses)
    % 간단한 지도 학습 모델 학습 (예시로 SVM 사용)
    model = fitcecoc(data.X, data.Y); % 예시로 다중 클래스 SVM 사용
end
%%
function model = update_supervised_model(model, new_data)
    % 기존 지도 학습 모델 업데이트
    disp('Updating supervised model (dummy operation)...');
    model = model; % 실제 모델 업데이트 대신 모델을 그대로 반환
end
%%
function compare_opinions(model, clustered_hn, clustered_reddit)
    % 예시로서 실제로 분석하지는 않음
    disp('Comparing opinions (dummy operation)...');
    disp('Opinions on Hacker News:');
    disp('Opinions on Reddit:');
end
