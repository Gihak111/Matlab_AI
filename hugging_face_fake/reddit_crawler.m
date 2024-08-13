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
        reddit_data = [reddit_data; struct('title', title, 'link', link)];
    end
end