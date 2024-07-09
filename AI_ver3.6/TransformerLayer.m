classdef TransformerLayer < nnet.layer.Layer
    properties
        % Hyperparameters
        NumHeads
        Dim
        Depth
        FeedForwardSize
        Dropout
        WordEmbeddingLayer
    end
    
    properties (Learnable)
        % Learnable parameters
        Q
        K
        V
        WO
        FF1
        FF2
        LayerNorm1
        LayerNorm2
    end
    
    methods
        function layer = TransformerLayer(numHeads, dim, depth, feedForwardSize, dropout, wordEmbeddingLayer, name)
            layer.NumHeads = numHeads;
            layer.Dim = dim;
            layer.Depth = depth;
            layer.FeedForwardSize = feedForwardSize;
            layer.Dropout = dropout;
            layer.Name = name;
            layer.WordEmbeddingLayer = wordEmbeddingLayer;
            
            % Initialize learnable parameters
            scale = sqrt(dim / numHeads);
            layer.Q = dlarray(single(randn(dim, dim)) * scale);
            layer.K = dlarray(single(randn(dim, dim)) * scale);
            layer.V = dlarray(single(randn(dim, dim)) * scale);
            layer.WO = dlarray(single(randn(dim, dim)) * scale);
            layer.FF1 = dlarray(single(randn(feedForwardSize, dim)) * scale);
            layer.FF2 = dlarray(single(randn(dim, feedForwardSize)) * scale);
            
            % Initialize layer normalization layers
            layer.LayerNorm1 = layerNormalizationLayer();
            layer.LayerNorm2 = layerNormalizationLayer();
        end
        
        function Z = predict(layer, X)
            Z = multiHeadAttention(layer, X);
            Z = layerNormalization(layer, Z + X, layer.LayerNorm1);
            
            Z = positionwiseFF(layer, Z);
            Z = layerNormalization(layer, Z + X, layer.LayerNorm2);
        end
        
        function Z = multiHeadAttention(layer, X)
            % Q, K, V
            Q = X * layer.Q;
            K = X * layer.K;
            V = X * layer.V;
            
            % Scale dot-product attention
            W = softmax(Q * K' / sqrt(size(Q, 2)), 2);
            Z = W * V;
        end
        
        function Z = positionwiseFF(layer, X)
            Z = relu(X * layer.FF1) * layer.FF2;
        end
    end
end
