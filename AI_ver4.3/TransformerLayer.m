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
            
            obj.QueryWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.KeyWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.ValueWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.OutputWeights = dlarray(randn(embedDim, embedDim, 'single'));
        end
        
        function Z = predict(obj, X)
            Q = mtimes(X, obj.QueryWeights);
            K = mtimes(X, obj.KeyWeights);
            V = mtimes(X, obj.ValueWeights);

            Q = reshape(Q, [], obj.NumHeads, size(Q, 2) / obj.NumHeads);
            K = reshape(K, [], obj.NumHeads, size(K, 2) / obj.NumHeads);
            V = reshape(V, [], obj.NumHeads, size(V, 2) / obj.NumHeads);
            
            attentionScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(obj.EmbedDim / obj.NumHeads);
            attentionWeights = softmaxx(attentionScores);
            
            attentionOutput = pagemtimes(attentionWeights, V);
            
            attentionOutput = reshape(attentionOutput, [], size(attentionOutput, 3) * obj.NumHeads);
            
            Z = mtimes(attentionOutput, obj.OutputWeights);
        end
    end
end