function lexicon = loadSentimentLexicon()
    lexicon = containers.Map();
    lexicon('good') = 1;
    lexicon('happy') = 1;
    lexicon('positive') = 1;
    lexicon('excellent') = 2;
    lexicon('bad') = -1;
    lexicon('sad') = -1;
    lexicon('negative') = -1;
    lexicon('poor') = -2;
end