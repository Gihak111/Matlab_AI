% 주요 문장을 추출하여 요약하는 함수
function summary = summarize_text(text)
    % Tokenize the text into sentences
    sentences = splitSentences(text);
    numSentences = numel(sentences);
    
    % Simple heuristic: select the first and the last sentences as the summary
    if numSentences > 1
        summary = join([sentences(1); sentences(end)], ' ');
    else
        summary = text; % If only one sentence, return the original text
    end
end