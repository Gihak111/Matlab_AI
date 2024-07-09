classdef wordEmbeddingLayer < nnet.layer.Layer
    properties
        EmbeddingDim
        NumWords
        WordIndexMap % 새로운 속성 추가
    end
    
    properties (Learnable)
        EmbeddingMatrix
    end
    
    methods
        function obj = wordEmbeddingLayer(embeddingDim, numWords, name)
            % 기본 속성 설정
            obj.EmbeddingDim = embeddingDim;
            obj.NumWords = numWords;
            obj.WordIndexMap = containers.Map(); % WordIndexMap 초기화
            obj.Name = name; % Name 속성 설정
            
            % 임의의 값으로 EmbeddingMatrix 초기화
            obj.EmbeddingMatrix = dlarray(randn(embeddingDim, numWords, 'single'));
        end
        
        function Z = predict(obj, X)
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
            doc = tokenizedDocument(text);
            details = tokenDetails(doc);
            words = string(details.Token);
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
            outputSize = [obj.EmbeddingDim 1];
        end
    end
end
