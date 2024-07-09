function output = custom_dropout(input, dropoutRate)
    % Custom dropout function for dlarray inputs
    if dropoutRate > 0
        keepProbability = 1 - dropoutRate;
        mask = rand(size(input)) < keepProbability;
        scaleFactor = 1 / keepProbability;
        output = input .* mask * scaleFactor;
    else
        output = input;
    end
end