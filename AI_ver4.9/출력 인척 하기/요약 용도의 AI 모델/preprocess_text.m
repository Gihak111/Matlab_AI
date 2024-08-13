% 텍스트 전처리 및 토큰화 함수
function processed_text = preprocess_text(text)
    % 간단한 전처리: 소문자 변환, 구두점 제거 등
    processed_text = lower(text);
    processed_text = regexprep(processed_text, '[^\w\s]', '');
end