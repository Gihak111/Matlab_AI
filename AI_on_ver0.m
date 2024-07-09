function online_learning_unsupervised()
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
        
        % 데이터 전처리 및 클러스터링
        [clustered_hn, clustered_reddit, enc] = preprocess_and_cluster(hn_data, reddit_data, enc);
        
        % 결과 요약
        summarize_results(clustered_hn, clustered_reddit);
        
        % 훈련된 모델 또는 클러스터 업데이트
        if isempty(model)
            % 최초 모델 생성 및 훈련
            model = create_clustering_model(clustered_hn.bag, 3); % 예시로 3개의 클러스터 사용
        else
            % 기존 모델 업데이트
            model = update_clustering_model(model, clustered_hn.bag);
        end
        
        % 결과 비교 실행
        compare_opinions(model, clustered_hn, clustered_reddit);
    end
end

function [clustered_hn, clustered_reddit, enc] = preprocess_and_cluster(hn_data, reddit_data, enc)
    % Combine news and reddit data
    all_data = [hn_data; reddit_data];
    
    % Extract texts
    texts = {all_data.title}';
    
    % Create bag-of-words representation
    [bag, enc] = create_bag_of_words(texts, enc);
    
    % Cluster sequences
    numClusters = 3; % 예시로 3개의 클러스터 사용
    idx = kmeans(bag, numClusters);
    
    % Separate clustered data
    clustered_hn = hn_data(idx <= numClusters);
    clustered_reddit = reddit_data(idx > numClusters);
    
    % Add bag-of-words to clustered data
    clustered_hn.bag = bag(idx <= numClusters, :);
    clustered_reddit.bag = bag(idx > numClusters, :);
end

function [bag, enc] = create_bag_of_words(texts, enc)
    if isempty(enc)
        % Create word encoding if not provided
        tokenizer = tokenizedDocument(texts);
        enc = wordEncoding(tokenizer);
    end
    
    numDocs = numel(texts);
    bag = zeros(numDocs, enc.NumWords);
    
    for i = 1:numDocs
        docWords = lower(strsplit(texts{i}));
        for j = 1:numel(docWords)
            wordIndex = word2ind(enc, docWords{j});
            if ~isempty(wordIndex) && wordIndex > 0 % 인덱스가 유효한지 확인
                bag(i, wordIndex) = bag(i, wordIndex) + 1;
            end
        end
    end
end


function summarize_results(clustered_hn, clustered_reddit)
    disp('--- Summarized Results ---');
    
    % Summarize Hacker News
    hn_titles = {clustered_hn.title}';
    hn_links = {clustered_hn.link}';
    disp('Hacker News Top Posts:');
    for i = 1:length(hn_titles)
        disp(['Title: ' hn_titles{i}]);
        disp(['Link: ' hn_links{i}]);
    end
    
    % Summarize Reddit
    reddit_titles = {clustered_reddit.title}';
    reddit_links = {clustered_reddit.link}';
    disp('Reddit Top Posts:');
    for i = 1:length(reddit_titles)
        disp(['Title: ' reddit_titles{i}]);
        disp(['Link: ' reddit_links{i}]);
    end
end

function model = create_clustering_model(data, numClusters)
    model = kmeans(data, numClusters); % K-means 클러스터링 모델 생성
    % 다른 클러스터링 모델을 사용할 수도 있음
end

function model = update_clustering_model(model, new_data)
    % 예시로서 실제로 클러스터링 모델을 업데이트하지는 않음
    disp('Updating clustering model (dummy operation)...');
    model = model; % 실제 모델 업데이트 대신 모델을 그대로 반환
end

function compare_opinions(model, clustered_hn, clustered_reddit)
    % 예시로서 실제로 분석하지는 않음
    disp('Comparing opinions (dummy operation)...');
    disp('Opinions on Hacker News:');
    disp('Opinions on Reddit:');
end

function hn_data = hackernews_crawler()
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty';
    options = weboptions('Timeout', 30);
    story_ids = webread(url, options);
    hn_data = [];

    for i = 1:min(10, length(story_ids))
        story_url = strcat('https://hacker-news.firebaseio.com/v0/item/', num2str(story_ids(i)), '.json?print=pretty');
        story = webread(story_url, options);
        hn_data = [hn_data; struct('title', story.title, 'link', story.url)];
    end
end

function reddit_data = reddit_crawler()
    url = 'https://www.reddit.com/r/news/top/.json?limit=10';
    options = weboptions('UserAgent', 'MATLAB', 'Timeout', 30);
    data = webread(url, options);
    reddit_data = [];

    children = data.data.children;
    numChildren = length(children);
    
    for i = 1:numChildren
        post = children(i).data;
        reddit_data = [reddit_data; struct('title', post.title, 'link', post.url)];
    end
end
