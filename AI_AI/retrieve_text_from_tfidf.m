function title_text = retrieve_text_from_tfidf(data, issue_keywords)
    % Simplified text retrieval based on TF-IDF data
    [~, max_idx] = max(data);
    title_text = issue_keywords{max_idx};
end