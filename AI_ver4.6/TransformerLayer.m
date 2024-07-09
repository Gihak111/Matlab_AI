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
        function obj = TransformerLayer(numHeads, embedDim, numTransformers, intermediateDim, dropoutRate, wordEmbeddingLayerObj, name)
            obj.NumHeads = numHeads;
            obj.EmbedDim = embedDim;
            obj.NumTransformers = numTransformers;
            obj.IntermediateDim = intermediateDim;
            obj.DropoutRate = dropoutRate;
            obj.Name = char(name); % Ensure name is converted to a character array
            
            % Initialize learnable parameters
            obj.QueryWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.KeyWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.ValueWeights = dlarray(randn(embedDim, embedDim, 'single'));
            obj.OutputWeights = dlarray(randn(embedDim, embedDim, 'single'));
        end
        
       
            function Z = predict(obj, X)
                % Convert input to dlarray if it's not already
                if ~isa(X, 'dlarray')
                    X = dlarray(X);
                end
                
                batchSize = size(X, 4); % Batch size
                seqLength = size(X, 1); % Sequence length
        
                % Self-attention mechanism (simplified)
                Q = obj.QueryWeights * X;
                K = obj.KeyWeights * X;
                V = obj.ValueWeights * X;
        
                % Compute attention scores
                attentionScores = (Q' * K) / sqrt(obj.EmbedDim);
                attentionWeights = softmax(attentionScores, 'DataFormat', 'CB'); % Corrected data format
        
                % Apply attention to values
                attentionOutput = attentionWeights * V;
        
                % Apply dropout (custom function for dlarray)
                attentionOutput = custom_dropout(attentionOutput, obj.DropoutRate); 
        
                % Residual connection and layer normalization
                residualOutput = obj.OutputWeights * attentionOutput;
                
                % Add residual connection and apply layer normalization
                Z = dlarray(residualOutput + X, 'CB'); % Corrected data format
           end
        

    end
end