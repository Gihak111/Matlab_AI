% 인기 글 요약 및 반응 해석
function summarize_and_interpret(all_data)
    % 조회수나 점수를 기준으로 정렬
    [~, idx] = sort([all_data.score], 'descend');
    popular_article = all_data(idx(1));
    
    fprintf('Title: %s\n', popular_article.title);
    fprintf('Link: %s\n', popular_article.link);
    fprintf('Score: %d\n', popular_article.score);
    
    try
        text = webread(popular_article.link);
        summary = extractSummary(text);
        fprintf('Summary: %s\n', summary);
        
        reaction = interpret_reaction(summary);
        fprintf('Reaction: %s\n', reaction);
    catch
        fprintf('Summary: Unable to fetch article content.\n');
        fprintf('Reaction: N/A\n');
    end
end
