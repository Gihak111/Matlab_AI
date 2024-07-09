function issues_found = check_issues(data, issue_keywords)
    issues_found = {};
    
    % Dummy issue detection based on keywords
    for i = 1:size(data, 1)
        title_text = retrieve_text_from_tfidf(data(i, :), issue_keywords);
        if ~isempty(title_text)
            issues_found = [issues_found title_text];
        end
    end
end