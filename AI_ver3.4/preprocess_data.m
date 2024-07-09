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
    
    % Initialize X as a cell array
    X = cell(numSequences, 1);
    Y = categorical(labels);
    
    % Pad and fill X with sequences
    for i = 1:numSequences
        % Get sequence and truncate if longer than maxLength
        seq = sequences{i};
        seqLength = min(numel(seq), maxLength);
        
        % Convert sequence to cell array of strings
        X{i} = string(seq(1:seqLength));
    end
end
