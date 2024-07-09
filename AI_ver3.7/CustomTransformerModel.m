classdef CustomTransformerModel < handle
    properties
        WordEmbeddingLayer
        TransformerLayer
        GlobalAvgPooling
        FullyConnected
        Softmax
        ClassificationOutput
    end
    
    methods
        function obj = CustomTransformerModel(embeddingDim, numWords, numHeads, dim, depth, feedForwardSize, dropout)
            % Initialize layers
            obj.WordEmbeddingLayer = wordEmbeddingLayer(embeddingDim, numWords);
            obj.TransformerLayer = TransformerLayer(numHeads, dim, depth, feedForwardSize, dropout, obj.WordEmbeddingLayer, 'Transformer');
            obj.GlobalAvgPooling = globalAveragePooling1dLayer();
            obj.FullyConnected = fullyConnectedLayer(2);
            obj.Softmax = softmaxLayer();
            obj.ClassificationOutput = classificationLayer();
        end
        
        function Z = predict(obj, X)
            Z = obj.WordEmbeddingLayer.predict(X);
            Z = obj.TransformerLayer.predict(Z);
            Z = forward(obj.GlobalAvgPooling, Z);
            Z = forward(obj.FullyConnected, Z);
            Z = forward(obj.Softmax, Z);
            Z = predict(obj.ClassificationOutput, Z);
        end
        
        function [loss, gradients] = modelGradients(obj, X, Y)
            % Forward pass
            predictions = obj.predict(X);
            % Loss calculation
            loss = crossentropy(predictions, Y);
            % Gradients calculation
            gradients = dlgradient(loss, obj.LearnableParameters);
        end
    end
end
