function compare_opinions(hn_data, reddit_data)
    if nargin < 1
        % hn_data를 전달받지 않은 경우 처리할 내용
        hn_data = []; % 또는 다른 기본값 설정
    end
    
    if nargin < 2
        % reddit_data를 전달받지 않은 경우 처리할 내용
        reddit_data = []; % 또는 다른 기본값 설정
    end
    
    summarize_and_analyze(hn_data, reddit_data);
end
