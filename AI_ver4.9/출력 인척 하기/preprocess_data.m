function [X, Y, enc] = preprocess_data(hn_data, reddit_data)
    % Combine news and Reddit data
    all_data = [hn_data; reddit_data];
    
    % Extract texts and create labels
    texts = {all_data.title}'; % Extract titles from all data
    labels = [zeros(length(hn_data), 1); ones(length(reddit_data), 1)]; % Create labels: 0 for Hacker News, 1 for Reddit
    
    % Tokenize texts
    tokenizer = tokenizedDocument(texts); % Tokenize the texts
    enc = wordEncoding(tokenizer); % Create a word encoding from the tokenizer
    
    % Set max length for sequences
    maxLength = 81; % Set the maximum length of sequences
    embeddingDim = 128; % Set the dimension of word embeddings
    
    % Convert documents to sequences
    sequences = doc2sequence(enc, tokenizer); % Convert documents to sequences of word indices
    numSequences = length(sequences); % Get the number of sequences
    
    % Initialize cell array for padded sequences
    X = cell(numSequences, 1);
    
    for i = 1:numSequences
        % Determine the length of the current sequence
        seqLength = min(numel(sequences{i}), maxLength);
        % Initialize the padded sequence with zeros
        paddedSequence = zeros(maxLength, embeddingDim);
        
        for j = 1:seqLength
            wordIndex = sequences{i}(j);
            % Create an embedding for the current word (random initialization for example)
            wordEmbedding = rand(1, embeddingDim); % In practice, use a pre-trained embedding
            paddedSequence(j, :) = wordEmbedding;
        end
        
        % Assign the padded sequence to the cell array
        X{i} = paddedSequence;
    end
    
    % Convert labels to categorical row vectors and store in a cell array
    Y = cell(numSequences, 1);
    for i = 1:numSequences
        Y{i} = categorical(labels(i));
    end
end
