%{
function opinions = analyze_opinions(clustered_data)
    % 감정 분석 및 결과 반환
    opinions = {};

    if ~isempty(clustered_data)
        num_clusters = length(clustered_data);
        for c = 1:num_clusters
            cluster_opinion = analyze_cluster(clustered_data(c), c);
            opinions{c} = cluster_opinion;
        end
    else
        opinions = {'No data available.'};
    end
end
%}
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
                title = cluster.X(s).title;
                if contains(lower(title), issue_keywords)
                    disp(['    {' title '}']);
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