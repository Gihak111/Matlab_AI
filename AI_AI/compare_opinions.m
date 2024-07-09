%{
function compare_opinions(model, clustered_hn, clustered_reddit)
    disp('Comparing opinions...');

    % 예시로 사용할 더미 데이터 (실제 분석 결과 사용)
    hn_opinions = analyze_opinions(clustered_hn);
    reddit_opinions = analyze_opinions(clustered_reddit);

    % 출력하기
    disp('Opinions on Hacker News:');
    disp(hn_opinions);
    disp('Opinions on Reddit:');
    disp(reddit_opinions);
end




function compare_opinions(model, clustered_hn, clustered_reddit)
    disp('Comparing opinions (dummy operation)...');
    disp('Opinions on Hacker News:');
    disp_analyzed_opinions(clustered_hn);
    disp('Opinions on Reddit:');
    disp_analyzed_opinions(clustered_reddit);
end
%}
 function compare_opinions(model, clustered_hn, clustered_reddit)
    % 예시로서 실제로 분석하지는 않음
    disp('Comparing opinions (dummy operation)...');
    disp('Opinions on Hacker News:');
    disp('Opinions on Reddit:');
end