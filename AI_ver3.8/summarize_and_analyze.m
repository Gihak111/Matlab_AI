% 주요 함수: 글을 요약하고 감정을 분석하여 출력
function summarize_and_analyze(news_data, reddit_data)
    for i = 1:numel(news_data)
        fprintf('Title: %s\n', news_data(i).title);
        fprintf('Link: %s\n', news_data(i).link);
        % 여기에 웹 페이지의 전체 텍스트를 가져오는 코드를 추가할 수 있습니다
        
        % For demonstration, let's use the title as the text
        text = news_data(i).title;
        
        % Summarize text
        summary = summarize_text(text);
        fprintf('Summary: %s\n', summary);
        
        % Analyze sentiment
        sentiment = analyze_sentiment(text);
        fprintf('Sentiment: %s\n', sentiment);
        fprintf('\n');
    end
    
    for i = 1:numel(reddit_data)
        fprintf('Title: %s\n', reddit_data(i).title);
        fprintf('Link: %s\n', reddit_data(i).link);
        % 여기에 웹 페이지의 전체 텍스트를 가져오는 코드를 추가할 수 있습니다
        
        % For demonstration, let's use the title as the text
        text = reddit_data(i).title;
        
        % Summarize text
        summary = summarize_text(text);
        fprintf('Summary: %s\n', summary);
        
        % Analyze sentiment
        sentiment = analyze_sentiment(text);
        fprintf('Sentiment: %s\n', sentiment);
        fprintf('\n');
    end
end