% 단어 인덱스 생성 함수
function [word_index, index_word] = create_word_index(data)
    words = [];
    for i = 1:length(data)
        words = [words; split(preprocess_text(data(i).title))];
    end
    unique_words = unique(words);
    word_index = containers.Map('KeyType','char','ValueType','int32');
    index_word = containers.Map('KeyType','int32','ValueType','char');
    for i = 1:length(unique_words)
        word_index(unique_words{i}) = i;
        index_word(i) = unique_words{i};
    end
end