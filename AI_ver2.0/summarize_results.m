function summarize_results(clustered_hn, clustered_reddit, texts, enc)
    % Retrieve issue keywords
    issue_keywords = extract_issue_keywords(texts, enc);
    
    % Print identified issues per cluster
    disp('--- Identified Issues ---');
    
    % Summarize issues for Hacker News data
    if ~isempty(clustered_hn)
        for cluster_id = 1:max(clustered_hn.Y)
            idx = find(clustered_hn.Y == cluster_id);
            if ~isempty(idx)
                disp(['Cluster ' num2str(cluster_id) ': ' num2str(length(idx)) ' samples']);
                issues_found = check_issues(clustered_hn.X(idx, :), issue_keywords);
                if ~isempty(issues_found)
                    disp(['Issues identified:']);
                    for issue = issues_found
                        disp(['    {' issue '}']);
                    end
                else
                    disp('No specific issues identified.');
                end
            end
        end
    end
    
    % Summarize issues for Reddit data
    if ~isempty(clustered_reddit)
        for cluster_id = 1:max(clustered_reddit.Y)
            idx = find(clustered_reddit.Y == cluster_id);
            if ~isempty(idx)
                disp(['Cluster ' num2str(cluster_id) ': ' num2str(length(idx)) ' samples']);
                issues_found = check_issues(clustered_reddit.X(idx, :), issue_keywords);
                if ~isempty(issues_found)
                    disp(['Issues identified:']);
                    for issue = issues_found
                        disp(['    {' issue '}']);
                    end
                else
                    disp('No specific issues identified.');
                end
            end
        end
    end
    
    % Dummy operations for model update and comparison
    disp('--- Updating supervised model (dummy operation)...');
    disp('--- Comparing opinions (dummy operation)...');
    disp('Opinions on Hacker News:');
    disp('Opinions on Reddit:');
end
