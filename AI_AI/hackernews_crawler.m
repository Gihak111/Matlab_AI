function hn_data = hackernews_crawler_online()
    % URL 설정
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty';
    
    % 웹 옵션 설정
    options = weboptions('Timeout', 30);
    
    % 최상위 스토리 ID 가져오기 및 예외 처리
    try
        story_ids = webread(url, options);
        
        % 데이터 초기화
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
        
        % 데이터가 비어 있는 경우 처리
        if isempty(hn_data)
            disp('No Hacker News data available.');
        end
        
    catch ME
        % 예외 처리
        disp('Failed to fetch Hacker News data.');
        disp(['Error Message: ' ME.message]);
        hn_data = []; % 데이터를 가져오지 못한 경우 빈 배열 반환
    end

    result_hacknews_croll = hn_data
end
