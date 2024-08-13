% 시퀀스 변환 함수
function sequence = text_to_sequence(text, word_index)
    words = split(preprocess_text(text));
    sequence = zeros(1, numel(words));
    for i = 1:numel(words)
        if isKey(word_index, words{i})
            sequence(i) = word_index(words{i});
        else
            sequence(i) = word_index('UNK'); % 알 수 없는 단어 처리
        end
    end
end