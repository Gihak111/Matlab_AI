% Sentiment lexicon 로드
function lexicon = loadSentimentLexicon()
    % Simple sentiment lexicon
    lexicon = containers.Map();
    lexicon('good') = 1;
    lexicon('happy') = 1;
    lexicon('positive') = 1;
    lexicon('excellent') = 2;
    lexicon('bad') = -1;
    lexicon('sad') = -1;
    lexicon('negative') = -1;
    lexicon('poor') = -2;
    % 추가적인 단어를 여기에 추가할 수 있습니다
end