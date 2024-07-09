% 나이대 및 정치적 입장 분석 함수
function analyze_demographics(news_data, reddit_data)
    % 샘플 데이터를 기반으로 가정
    % 실제 데이터는 뉴스 및 Reddit 댓글에서 추출해야 함
    
    age_groups = {'18-24', '25-34', '35-44', '45-54', '55+'};
    political_views = {'Liberal', 'Conservative', 'Moderate'};
    
    % 랜덤하게 나이대와 정치적 입장을 할당
    num_samples = length(news_data) + length(reddit_data);
    ages = age_groups(randi([1 length(age_groups)], [num_samples 1]));
    views = political_views(randi([1 length(political_views)], [num_samples 1]));
    
    % 연령대 및 정치적 입장별 데이터 분석
    for i = 1:length(age_groups)
        fprintf('Age Group: %s\n', age_groups{i});
        idx = strcmp(ages, age_groups{i});
        
        for j = 1:length(political_views)
            fprintf('Political View: %s\n', political_views{j});
            sub_idx = idx & strcmp(views, political_views{j});
            
            if sum(sub_idx) > 0
                subset = [news_data; reddit_data](sub_idx);
                
                for k = 1:length(subset)
                    text = subset(k).title;
                    summary = summarize_text(text);
                    sentiment = analyze_sentiment(text);
                    fprintf('Title: %s\n', subset(k).title);
                    fprintf('Summary: %s\n', summary);
                    fprintf('Sentiment: %s\n', sentiment);
                    fprintf('\n');
                end
            else
                fprintf('No data available.\n\n');
            end
        end
    end
end
