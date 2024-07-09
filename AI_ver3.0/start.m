% 메인 스크립트
hn_data = hackernews_crawler(); % Hacker News 데이터 크롤링
reddit_data = reddit_crawler(); % Reddit 데이터 크롤링

% 데이터 전처리
[X, Y, enc] = preprocess_data(hn_data, reddit_data); % 데이터 전처리

% 인기 글 요약 및 반응 해석
summarize_and_interpret([hn_data; reddit_data]);