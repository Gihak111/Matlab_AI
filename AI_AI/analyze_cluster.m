function cluster_opinion = analyze_cluster(cluster, cluster_index)
    % 각 클러스터의 감정 분석
    if isfield(cluster, 'X') && ~isempty(cluster.X)
        cluster_opinion = sprintf('Cluster %d:', cluster_index);
        issues_found = false;
        for s = 1:length(cluster.X)
            title = cluster.X(s).title;
            if contains(lower(title), issue_keywords)
                cluster_opinion = [cluster_opinion sprintf('\n    {%s}', title)];
                issues_found = true;
            end
        end
        if ~issues_found
            cluster_opinion = [cluster_opinion ' No specific issues identified.'];
        end
    else
        cluster_opinion = 'Empty cluster.';
    end
end