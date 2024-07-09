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
            disp(size(X))
            disp(size(obj.QueryWeights))
            disp(size(obj.KeyWeights))
            disp(size(obj.ValueWeights))
        
            % Pad X with zeros to make it 128x128
            padded_X = [X, zeros(size(X, 1), 109, 'single')]; % Assuming 109 columns of padding
        
            % Perform matrix multiplication with weights
            Q = mtimes(padded_X, obj.QueryWeights);
            K = mtimes(padded_X, obj.KeyWeights);
            V = mtimes(padded_X, obj.ValueWeights);
        
            % Split into multiple heads (if necessary)
            Q = reshape(Q, size(Q, 1), obj.NumHeads, []);
            K = reshape(K, size(K, 1), obj.NumHeads, []);
            V = reshape(V, size(V, 1), obj.NumHeads, []);
        
            % Scaled dot-product attention
            attentionScores = pagemtimes(Q, 'none', K, 'transpose') / sqrt(obj.EmbedDim / obj.NumHeads);
        
            % Check the size of attentionScores
            disp(size(attentionScores));
        
            % Compute softmax using MATLAB's built-in softmax function
            % Adjust the dimension along which softmax is applied as per the structure of attentionScores
            attentionWeights = softmaxx(attentionScores);; % Apply softmax along the third dimension
        
            % Compute attention output
            attentionOutput = pagemtimes(attentionWeights, V);
        
            % Concatenate heads
            attentionOutput = reshape(attentionOutput, size(attentionOutput, 1), []);
        
            % Apply output weights
            Z = mtimes(attentionOutput, obj.OutputWeights);
            disp("End of predict")
        end


    end
end
