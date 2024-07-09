% 데이터 전처리 함수
function [X, Y, enc] = preprocess_data(news_data, reddit_data)
    % Combine news and reddit data
    all_data = [news_data; reddit_data];
    
    % Extract texts and create labels
    texts = {all_data.title}';
    labels = [zeros(length(news_data), 1); ones(length(reddit_data), 1)];
    
    % Tokenize texts
    tokenizer = tokenizedDocument(texts);
    enc = wordEncoding(tokenizer);
    
    % Ensure sequences length does not exceed maxLength
    maxLength = 100;
    sequences = doc2sequence(enc, tokenizer);
    numSequences = length(sequences);
    
    % Initialize X and Y
    X = zeros(maxLength, numSequences, 'uint32'); % 크기 수정
    Y = categorical(labels);
    
    % Pad and fill X with sequences
    for i = 1:numSequences
        seqLength = min(numel(sequences{i}), maxLength);
        X(1:seqLength, i) = sequences{i}(1:seqLength);
    end
end