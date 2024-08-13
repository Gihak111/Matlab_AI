% 시퀀스 디코딩 함수
function summary = decode_sequence(encoded_state, dec_model, word_index, max_len)
    index_word = values(word_index);
    num_words = length(index_word);
    start_idx = word_index('START');
    end_idx = word_index('END');
    
    summary = '';
    current_word = start_idx;
    for i = 1:max_len
        % 디코더 모델 예측
        [output, encoded_state] = dec_model.predict({current_word, encoded_state});
        
        % 다음 단어 예측
        [~, idx] = max(output);
        next_word = idx;
        
        % 종료 조건 확인
        if next_word == end_idx
            break;
        end
        
        % 단어 추가
        if next_word ~= start_idx
            summary = [summary ' ' index_word{next_word}];
        end
        
        % 다음 입력 설정
        current_word = next_word;
    end
end