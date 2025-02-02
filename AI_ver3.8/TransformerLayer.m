classdef TransformerLayer < nnet.layer.Layer
    properties
        NumHeads
        EmbedDim
        NumTransformers
        IntermediateDim
        DropoutRate
    end
    
    properties (Learnable)
        QueryWeights
        KeyWeights
        ValueWeights
        OutputWeights
    end
    
    methods
        function obj = TransformerLayer(numHeads, embedDim, numTransformers, intermediateDim, dropoutRate, tokenizer, name)
            obj.NumHeads = numHeads;
            obj.EmbedDim = embedDim;
            obj.NumTransformers = numTransformers;
            obj.IntermediateDim = intermediateDim;
            obj.DropoutRate = dropoutRate;
            obj.Name = name;
            
            % Initialize learnable parameters
            obj.QueryWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.KeyWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.ValueWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.OutputWeights = dlarray(randn(embedDim, embedDim, 'single'));
        end
        
            function Z = predict(obj, X)
            % Multi-head self-attention mechanism
            Q = X .* obj.QueryWeights;
            K = X .* obj.KeyWeights;
            V = X .* obj.ValueWeights;
    
            % Split into multiple heads
            Q = reshape(Q, size(Q, 1), obj.NumHeads, []);
            K = reshape(K, size(K, 1), obj.NumHeads, []);
            V = reshape(V, size(V, 1), obj.NumHeads, []);
    
            % Scaled dot-product attention
            attentionScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(obj.EmbedDim / obj.NumHeads);

            % Print the size of attentionScores
            disp('Size of attentionScores before permute:');
            disp(size(attentionScores));

            % Reshape attentionScores
            attentionScores = permute(attentionScores, [3 1 2]);

            % Print the size of attentionScores after permute
            disp('Size of attentionScores after permute:');
            disp(size(attentionScores));

            % Compute softmax
            attentionWeights = softmax(attentionScores, 'DataFormat', 'SSB');

            % Reshape attentionWeights back to original form
            attentionWeights = permute(attentionWeights, [2 3 1]);

            % Compute attention output
            attentionOutput = pagemtimes(attentionWeights, V);



    
            % Concatenate heads
            %attentionOutput = reshape(attentionOutput, size(attentionOutput, 1), []);
    
            % Apply output weights
            %Z = attentionOutput * obj.OutputWeights;
        end


    end
end
