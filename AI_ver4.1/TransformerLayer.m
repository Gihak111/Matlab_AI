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
            % Convert X to dlarray with 'SSCB' format (SequenceLength, MiniBatchSize, Channels)
            X = dlarray(X, 'SSCB');  % Assuming X has size [82, 1, 1]
            
            % Perform matrix multiplication with weights
            Q = mtimes(X, obj.QueryWeights);
            K = mtimes(X, obj.KeyWeights);
            V = mtimes(X, obj.ValueWeights);
        
            % Split into multiple heads (if necessary)
            Q = reshape(Q, [size(Q, 1), obj.NumHeads, size(Q, 2)/obj.NumHeads]);
            K = reshape(K, [size(K, 1), obj.NumHeads, size(K, 2)/obj.NumHeads]);
            V = reshape(V, [size(V, 1), obj.NumHeads, size(V, 2)/obj.NumHeads]);
        
            % Scaled dot-product attention
            attentionScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(obj.EmbedDim / obj.NumHeads);
        
            % Compute softmax using MATLAB's built-in softmax function
            attentionWeights = softmax(attentionScores, 3);  % Apply softmax along the third dimension
        
            % Compute attention output
            attentionOutput = pagemtimes(attentionWeights, V);
        
            % Concatenate heads
            attentionOutput = reshape(attentionOutput, [size(attentionOutput, 1), size(attentionOutput, 2) * size(attentionOutput, 3)]);
        
            % Apply output weights
            Z = mtimes(attentionOutput, obj.OutputWeights);
        end
    end
end
