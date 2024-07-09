function hasIssue = hasIssueKeywords(tokens)
    issue_keywords = ["2022-07-08", "2023", "Ants", "About"]; % Define issue keywords
    
    % Check if any of the tokens match issue keywords
    hasIssue = any(ismember(tokens, issue_keywords));
end