function summarize_and_analyze(news_data, reddit_data)
    for i = 1:numel(news_data)
        fprintf('Title: %s\n', news_data(i).title);
        fprintf('Link: %s\n', news_data(i).link);
        
        text = news_data(i).title;
        
        summary = summarize_text(text);
        fprintf('Summary: %s\n', summary);
        
        sentiment = analyze_sentiment(text);
        fprintf('Sentiment: %s\n', sentiment);
        fprintf('\n');
    end
    
    for i = 1:numel(reddit_data)
        fprintf('Title: %s\n', reddit_data(i).title);
        fprintf('Link: %s\n', reddit_data(i).link);
        
        text = reddit_data(i).title;
        
        summary = summarize_text(text);
        fprintf('Summary: %s\n', summary);
        
        sentiment = analyze_sentiment(text);
        fprintf('Sentiment: %s\n', sentiment);
        fprintf('\n');
    end
end