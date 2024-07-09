function [X, Y, enc] = preprocess_data(hn_data, reddit_data)
    all_data = [hn_data; reddit_data];
    texts = {all_data.title}';
    labels = [zeros(length(hn_data), 1); ones(length(reddit_data), 1)];
    
    tokenizer = tokenizedDocument(texts);
    enc = wordEncoding(tokenizer);
    
    maxLength = 100;
    sequences = doc2sequence(enc, tokenizer);
    numSequences = length(sequences);
    
    for i = 1:numSequences
        seqLength = min(numel(sequences{i}), maxLength);
        sequences{i} = padarray(sequences{i}(1:seqLength), [maxLength-seqLength 0], 'post');
    end
    
    X = sequences;
    Y = categorical(labels);
end