function summary = summarize_text(text)
    sentences = splitSentences(text);
    numSentences = numel(sentences);
    if numSentences > 1
        summary = join([sentences(1); sentences(end)], ' ');
    else
        summary = text;
    end
end