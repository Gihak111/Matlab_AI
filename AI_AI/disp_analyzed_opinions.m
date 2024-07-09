function disp_analyzed_opinions(clustered_data)
    if isempty(clustered_data)
        disp('No data available.');
        return;
    end
    
    for c = 1:length(clustered_data)
        cluster = clustered_data(c);
        disp(['Cluster ' num2str(c) ':']);
        
        if isfield(cluster, 'X') && ~isempty(cluster.X)
            issues_found = false;
            for s = 1:length(cluster.X)
                % Tokenize title assuming it's in a cell array
                title_tokens = tokenizeTitle(cluster.X(s).title{1});
                
                % Check for issue keywords
                if hasIssueKeywords(title_tokens)
                    disp(['    {' cluster.X(s).title{1} '}']);
                    issues_found = true;
                end
            end
            if ~issues_found
                disp("    No specific issues identified.");
            end
        else
            disp("    Empty cluster.");
        end
    end
end