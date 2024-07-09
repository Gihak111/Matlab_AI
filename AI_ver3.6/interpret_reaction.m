% 반응 해석 함수
function reaction = interpret_reaction(summary)
    % 간단한 반응 해석 예시
    positiveWords = ["good", "great", "excellent", "positive", "beneficial"];
    negativeWords = ["bad", "poor", "negative", "harmful", "awful"];
    
    positiveCount = sum(contains(lower(summary), positiveWords));
    negativeCount = sum(contains(lower(summary), negativeWords));
    
    if positiveCount > negativeCount
        reaction = "Overall positive reaction.";
    elseif negativeCount > positiveCount
        reaction = "Overall negative reaction.";
    else
        reaction = "Neutral reaction.";
    end
end