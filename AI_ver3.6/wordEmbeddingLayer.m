classdef wordEmbeddingLayer < nnet.layer.Layer
    properties
        EmbeddingMatrix
        WordIndexMap
        EmbeddingDim
        Tokenizer
    end
    
    methods
        function obj = wordEmbeddingLayer(embeddingDim, numWords)
            obj.EmbeddingDim = embeddingDim;
            obj.EmbeddingMatrix = dlarray(randn(embeddingDim, numWords, 'single'));
            obj.WordIndexMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');
            obj.Tokenizer = [];
        end
        
        function Z = predict(obj, X)
            % X는 길이가 maxLength인 벡터
            numSequences = size(X, 2);
            Z = zeros(obj.EmbeddingDim, numSequences, 'like', obj.EmbeddingMatrix);
            
            for i = 1:numSequences
                words = X(:, i);
                for j = 1:length(words)
                    if words(j) ~= 0
                        word = obj.tokenize(num2str(words(j)));
                        Z(:, i) = Z(:, i) + obj.embed(word);
                    end
                end
            end
            
            Z = dlarray(Z);
        end
        
        function words = tokenize(obj, text)
            if isempty(obj.Tokenizer)
                obj.Tokenizer = tokenizedDocument(text);
            else
                obj.Tokenizer.TextData = text;
            end
            words = string(obj.Tokenizer.TokenList);
        end

        function emb = embed(obj, word)
            if isKey(obj.WordIndexMap, word)
                idx = obj.WordIndexMap(word);
            else
                idx = length(obj.WordIndexMap) + 1;
                obj.WordIndexMap(word) = idx;
            end
            emb = obj.EmbeddingMatrix(:, idx);
        end
        
        function outputSize = getOutputSize(~)
            outputSize = [128 1];
        end
    end
end
