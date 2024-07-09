function enc = fit_tfidf(texts)
    % Tokenize the texts into words
    documents = tokenizedDocument(texts);
    
    % Create a vocabulary list
    vocab = unique([documents.Vocabulary]);
    
    % Number of documents
    numDocs = length(documents);
    
    % Initialize term frequency (TF) matrix
    tf = zeros(numDocs, length(vocab));
    
    for i = 1:numDocs
        wordsInDoc = documents(i).Vocabulary;
        for j = 1:length(vocab)
            tf(i, j) = sum(strcmp(wordsInDoc, vocab{j}));
        end
    end
    
    % Compute document frequency (DF) vector
    df = sum(tf > 0, 1);
    
    % Compute inverse document frequency (IDF) vector
    idf = log(numDocs ./ (df + 1)) + 1;
    
    % Store the vocabulary and IDF values in the encoder struct
    enc = struct('Vocabulary', vocab, 'IDF', idf);
end