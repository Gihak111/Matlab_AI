function [X, Y, enc] = preprocess_data(hn_data, reddit_data)
    % Combine news and reddit data
    all_data = [hn_data; reddit_data];
    
    % Extract texts and create labels
    texts = {all_data.title}';
    labels = [zeros(length(hn_data), 1); ones(length(reddit_data), 1)];
    
    % Tokenize texts
    tokenizer = tokenizedDocument(texts);
    enc = wordEncoding(tokenizer);
    
    % Ensure sequences length does not exceed maxLength
    maxLength = 100; % 예시 값, 데이터에 따라 조정 필요
    sequences = doc2sequence(enc, tokenizer);
    numSequences = length(sequences);
    
    % Pad sequences if necessary
    for i = 1:numSequences
        seqLength = min(numel(sequences{i}), maxLength);
        sequences{i} = padarray(sequences{i}(1:seqLength), [maxLength-seqLength 0], 'post');
    end
    
    % Convert sequences to dlarray for compatibility with dlframework
    X = sequences;
    Y = categorical(labels);
end
