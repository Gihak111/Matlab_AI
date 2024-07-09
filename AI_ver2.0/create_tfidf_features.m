function [features, enc] = create_tfidf_features(texts, enc)
    % Create TF-IDF features
    if isempty(enc)
        enc = fit_tfidf(texts);
    end
    features = encode_tfidf(enc, texts);
end