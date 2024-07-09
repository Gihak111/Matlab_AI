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
        function obj = TransformerLayer(numHeads, embedDim, numTransformers, intermediateDim, dropoutRate, name)
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
            % Perform matrix multiplication with weights
            Q = mtimes(X, obj.QueryWeights);
            K = mtimes(X, obj.KeyWeights);
            V = mtimes(X, obj.ValueWeights);

            % Split into multiple heads (if necessary)
            Q = reshape(Q, size(Q, 1), obj.NumHeads, []);
            K = reshape(K, size(K, 1), obj.NumHeads, []);
            V = reshape(V, size(V, 1), obj.NumHeads, []);

            % Scaled dot-product attention
            attentionScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(obj.EmbedDim / obj.NumHeads);

            % Compute softmax
            attentionWeights = softmax(attentionScores, 3);

            % Compute attention output
            attentionOutput = pagemtimes(attentionWeights, V);

            % Concatenate heads
            attentionOutput = reshape(attentionOutput, size(attentionOutput, 1), []);

            % Apply output weights
            Z = mtimes(attentionOutput, obj.OutputWeights);
        end
    end
end