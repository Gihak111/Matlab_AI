% Hacker News 크롤러
function hn_data = hackernews_crawler()
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty';
    options = weboptions('Timeout', 30);
    story_ids = webread(url, options);
    hn_data = [];
    
    for i = 1:min(10, length(story_ids))
        story_url = strcat('https://hacker-news.firebaseio.com/v0/item/', num2str(story_ids(i)), '.json?print=pretty');
        story = webread(story_url, options);
        title = story.title;
        link = story.url;
        score = story.score; % 인기 점수
        hn_data = [hn_data; struct('title', title, 'link', link, 'score', score)];
    end
end