function tfidfMatrix = encode_tfidf(enc, texts)
    % Tokenize the texts into words
    documents = tokenizedDocument(texts);
    
    % Initialize term frequency (TF) matrix
    numDocs = length(documents);
    vocab = enc.Vocabulary;
    tf = zeros(numDocs, length(vocab));
    
    for i = 1:numDocs
        wordsInDoc = documents(i).Vocabulary;
        for j = 1:length(vocab)
            tf(i, j) = sum(strcmp(wordsInDoc, vocab{j}));
        end
    end
    
    % Compute TF-IDF matrix
    tfidfMatrix = tf .* enc.IDF;
end