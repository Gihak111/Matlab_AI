% 기사 내용 요약 (간단한 요약 방법 사용)
function summary = extractSummary(text)
    % 간단히 첫 n문장을 요약으로 사용하는 예시
    n = 2;
    sentences = splitSentences(text);
    summary = strjoin(sentences(1:min(n, numel(sentences))), ' ');
end