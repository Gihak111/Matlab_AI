%{
function reddit_data = reddit_crawler_online()
    % URL 설정
    url = 'https://www.reddit.com/r/news/top/.json?limit=10';
    
    % 웹 옵션 설정
    options = weboptions('UserAgent', 'MATLAB', 'Timeout', 30);
    
    % Reddit 데이터 가져오기 및 예외 처리
    try
        data = webread(url, options);
        
        % 데이터 초기화
        reddit_data = [];
        
        % children에서 제목과 링크 추출
        children = data.data.children;
        numChildren = length(children);
        
        % 데이터 추출
        for i = 1:numChildren
            post = children(i).data;
            title = post.title;
            link = post.url;
            reddit_data = [reddit_data; struct('title', title, 'link', link)];
        end
        
        % 데이터가 비어 있는 경우 처리
        if isempty(reddit_data)
            disp('No Reddit data available.');
        end
        
    catch ME
        % 예외 처리
        disp('Failed to fetch Reddit data.');
        disp(['Error Message: ' ME.message]);
        reddit_data = []; % 데이터를 가져오지 못한 경우 빈 배열 반환
    end
end
%}
function reddit_data = reddit_crawler2()
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
        
        % 데이터가 비어 있는 경우 처리
        if isempty(reddit_data)
            disp('No Reddit data available.');
        end
        
    catch ME
        disp('Failed to fetch Reddit data.');
        disp(['Error Message: ' ME.message]);
        reddit_data = []; % 데이터를 가져오지 못한 경우 빈 배열 반환
    end

    result_reddit_croll = reddit_data; % reddit_data를 직접 할당
end
