% 감정 분석을 수행하는 함수
function sentiment = analyze_sentiment(text)
    % Load pre-built sentiment lexicon
    lexicon = loadSentimentLexicon();
    
    % Tokenize the text into words
    doc = tokenizedDocument(text);
    words = tokenDetails(doc);
    
    % Initialize sentiment score
    sentimentScore = 0;
    
    % Calculate sentiment score based on lexicon
    for i = 1:numel(words.Token)
        word = words.Token{i};
        if isKey(lexicon, word)
            sentimentScore = sentimentScore + lexicon(word);
        end
    end
    
    % Determine sentiment
    if sentimentScore > 0
        sentiment = 'Positive';
    elseif sentimentScore < 0
        sentiment = 'Negative';
    else
        sentiment = 'Neutral';
    end
end