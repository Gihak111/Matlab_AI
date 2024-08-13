% Seq2Seq 모델 학습 및 요약 생성 함수
function summary = generate_summary(text, enc_model, dec_model, word_index, max_len)
    % 텍스트 전처리 및 시퀀스 변환
    input_seq = text_to_sequence(text, word_index);
    
    % LSTM 인코더를 사용하여 입력 시퀀스 처리
    encoded_state = enc_model.predict(input_seq);
    
    % 디코더를 사용하여 요약 생성
    summary = decode_sequence(encoded_state, dec_model, word_index, max_len);
end