% 레딧 크롤러
function reddit_data = reddit_crawler()
    url = 'https://www.reddit.com/r/news/top/.json?limit=10';
    options = weboptions('UserAgent', 'MATLAB', 'Timeout', 30);
    data = webread(url, options);
    reddit_data = [];
    
    children = data.data.children;
    numChildren = length(children);
    
    for i = 1:numChildren
        post = children(i).data;
        title = post.title;
        link = post.url;
        score = post.score; % 인기 점수
        reddit_data = [reddit_data; struct('title', title, 'link', link, 'score', score)];
    end
end