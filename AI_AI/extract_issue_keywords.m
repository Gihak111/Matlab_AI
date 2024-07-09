function issue_keywords = extract_issue_keywords(texts, enc)
    % Convert texts to tokenized documents
    documents = tokenizedDocument(texts);
    
    % Calculate TF-IDF features
    tfidfMatrix = encode_tfidf(enc, texts);
    
    % Summarize TF-IDF matrix to get important words
    min_tf_idf = 0.1; % 최소 TF-IDF 임계값 설정
    importantWords = cell(1, size(tfidfMatrix, 1));
    
    for i = 1:size(tfidfMatrix, 1)
        [~, idx] = sort(tfidfMatrix(i, :), 'descend');
        topWords = enc.Vocabulary(idx);
        topWords = topWords(tfidfMatrix(i, idx) >= min_tf_idf); % 임계값 이상의 단어만 선택
        importantWords{i} = topWords;
    end
    
    % Select unique and relevant keywords
    issue_keywords = unique([importantWords{:}]);
end