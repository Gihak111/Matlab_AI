% 메인 스크립트
hn_data = hackernews_crawler(); % Hacker News 데이터 크롤링
reddit_data = reddit_crawler(); % Reddit 데이터 크롤링

summarize_and_analyze(hn_data, reddit_data);