% 데이터 수집
hn_data = hackernews_crawler();
reddit_data = reddit_crawler();

% Python 패키지 설치
%system('pip install transformers');

% 요약 생성
hn_summaries = summarize_text(hn_data);
reddit_summaries = summarize_text(reddit_data);

% 결과 출력
fprintf('\n=== Hacker News Top Stories ===\n');
for i = 1:length(hn_data)
    fprintf('Title: %s\n', hn_data(i).title);
    fprintf('Link: %s\n', hn_data(i).link);
    fprintf('Summary: %s\n', hn_summaries{i});
    fprintf('----------------------------------\n');
end

fprintf('\n=== Reddit Top News ===\n');
for i = 1:length(reddit_data)
    fprintf('Title: %s\n', reddit_data(i).title);
    fprintf('Link: %s\n', reddit_data(i).link);
    fprintf('Summary: %s\n', reddit_summaries{i});
    fprintf('----------------------------------\n');
end

% 분석 추가 (예를 들어, 요약된 텍스트 간 비교)
% 여기서 추가적인 분석을 수행할 수 있습니다.
