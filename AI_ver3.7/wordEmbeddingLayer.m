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
        
        function Z = predictAndUpdateState(obj, X)
            numSequences = numel(X);
            Z = zeros(obj.EmbeddingDim, numSequences, 'like', obj.EmbeddingMatrix);

            for i = 1:numSequences
                words = X{i};
                for j = 1:length(words)
                    Z(:, i) = Z(:, i) + obj.embed(words{j});
                end
            end

            Z = dlarray(Z);
        end
        
        function Z = predict(obj, dlnet, X)
            %{
            numSequences = numel(X);
            words = cell(numSequences, 1);
            for i = 1:numSequences
                words{i} = obj.tokenize(X{i});
            end

            Z = zeros(obj.EmbeddingDim, numSequences, 'like', obj.EmbeddingMatrix);
            for i = 1:numSequences
                for j = 1:length(words{i})
                    Z(:, i) = Z(:, i) + obj.embed(words{i}(j));
                end
            end

            Z = dlarray(Z);

            Z = predictAndUpdateState(obj, Z);

            Z = predict(dlnet, Z);
            %}
            Z = predictAndUpdateState(obj, obj.EmbeddingMatrix);
            Z = predict(dlnet, Z);


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
