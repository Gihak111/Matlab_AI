classdef wordEmbeddingLayer < nnet.layer.Layer
    properties
        EmbeddingMatrix
        WordIndexMap
        EmbeddingDim
    end
    
    methods
        function obj = wordEmbeddingLayer(embeddingDim, numWords)
            obj.EmbeddingDim = embeddingDim;
            obj.EmbeddingMatrix = dlarray(randn(embeddingDim, numWords, 'single'));
            obj.WordIndexMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');
        end
        
        function Z = predict(obj, X)
            % X는 셀 배열로, 각 요소는 문자열입니다.
            disp('Value of X in predict function:');
            disp(X);

            % 전처리된 데이터의 크기 확인
            numSequences = numel(X); % 각 요소의 개수 반환

            % 결과를 저장할 Z 초기화
            Z = zeros(obj.EmbeddingDim, numSequences, 'like', obj.EmbeddingMatrix);

            % 각 시퀀스에 대해 embed 함수를 적용하여 Z에 더하기
            for i = 1:numSequences
                words = X{i}; % 각 셀 요소에 대해 문자열 배열에 접근
                for j = 1:length(words)
                    Z(:, i) = Z(:, i) + obj.embed(words{j});
                end
            end

            % dlarray로 변환하여 반환
            Z = dlarray(Z);
        end


        function emb = embed(obj, word)
            % Get word index or assign new index
            if isKey(obj.WordIndexMap, word)
                idx = obj.WordIndexMap(word);
            else
                idx = length(obj.WordIndexMap) + 1;
                obj.WordIndexMap(word) = idx;
            end
            
            % Return embedding vector
            emb = obj.EmbeddingMatrix(:, idx);
        end
        
        function outputSize = getOutputSize(~)
            outputSize = [128 1]; % 고정된 출력 크기
        end
    end
end
