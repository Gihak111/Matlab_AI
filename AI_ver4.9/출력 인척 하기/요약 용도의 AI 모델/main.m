% 메인 스크립트
% 여기서는 간단한 예제로, 미리 학습된 모델이 있다고 가정합니다.
% 미리 학습된 모델 대신, 실제 학습 및 모델 정의 과정을 포함시키려면 해당 부분을 구현해야 합니다.

% 크롤링 데이터 가져오기
hn_data = hackernews_crawler();
reddit_data = reddit_crawler();

% 최대 요약 길이 설정
max_len = 50;

% 단어 인덱스 맵 생성 (임의로 생성한 예시)
word_index = containers.Map();
word_index('START') = 1;
word_index('END') = 2;
word_index('UNK') = 3;
word_index('title') = 4;
word_index('link') = 5;

% 예시로 사용할 인코더 및 디코더 모델 (실제 학습된 모델 대신)
enc_model = []; % 인코더 모델 정의
dec_model = []; % 디코더 모델 정의

% 결과 출력
fprintf('Hacker News Summaries and Analysis:\n');
for i = 1:length(hn_data)
    summary = generate_summary(hn_data(i).title, enc_model, dec_model, word_index, max_len);
    fprintf('Title: %s\n', hn_data(i).title);
    fprintf('Link: %s\n', hn_data(i).link);
    fprintf('Summary: %s\n\n', summary);
end

fprintf('Reddit Summaries and Analysis:\n');
for i = 1:length(reddit_data)
    summary = generate_summary(reddit_data(i).title, enc_model, dec_model, word_index, max_len);
    fprintf('Title: %s\n', reddit_data(i).title);
    fprintf('Link: %s\n', reddit_data(i).link);
    fprintf('Summary: %s\n\n', summary);
end