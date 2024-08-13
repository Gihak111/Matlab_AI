classdef SequenceClassificationDispatcher < nnet.internal.cnn.dispatcher.BuiltInSequenceDispatcher
    % SequenceClassificationDispatcher   Dispatch time series data for
    % classification problems one mini batch at a time from a set of time
    % series
    %
    % Input data    - cell array of time series objects, where the length
    %               of the cell array is the number of observations. 
    %               Input predictors are numObs-by-1 cell arrays which
    %               contain sequences of size: DataSize-by-S
    %               Input responses are either:
    %                   - numObs-by-1 categorical arrays
    %                   - numObs-by-1 cell arrays which contain categorical
    %                   arrays of length 1-by-S. S here must match the S of
    %                   the corresponding predictor.
    %               
    % Output data   - numeric arrays with the following dimensions:
    %               Output predictors:
    %                   - DataSize-by-MiniBatchSize-by-S
    %               Output responses are either:
    %                   - ResponseSize-by-MiniBatchSize
    %                   - ResponseSize-by-MiniBatchSize-by-S
    
    %   Copyright 2017-2022 The MathWorks, Inc.
    
    properties (SetAccess = private)
        % DataSize (int)   Number of dimensions per time step of the input
        % data (D)
        DataSize
        
        % ResponseSize (int)   Number of classes in the response
        ResponseSize
        
        % NumObservations (int)   Number of observations in the data set
        NumObservations
        
        % SequenceLength   Strategy to determine the length of the
        % sequences used per mini-batch. Options are:
        %       - 'longest' to pad all sequences in a batch to the length
        %       of the longest sequence
        %       - 'shortest' to truncate all sequences in a batch to the
        %       length of the shortest sequence
        %       -  Positive integer - Pad sequences to the have same length
        %       as the longest sequence, then split into smaller sequences
        %       of the specified length. If splitting occurs, then the
        %       function creates extra mini-batches
        SequenceLength
        
        % DispatcherFormat (char)   Format of the response. Either:
        %       'seq2one'  : classify the entire sequence. Response is
        %       numObs-by-1 categorical array.
        %       'seq2seq'  : classify each time step. Response is
        %       numObs-by-1 cell array, containing 1-by-SequenceLength
        %       categorical arrays.        
        %       'predict'  : response is empty
        DispatcherFormat
        
        % NextStrategy (nnet.internal.cnn.sequence.NextStrategy)   Strategy
        % class which determines how mini-batches are prepared, based on:
        %       - DispatcherFormat
        %       - SequenceLength
        %       - readStrategy
        NextStrategy
        
        % ResponseMetaData
        ResponseMetaData = nnet.internal.cnn.response.ResponseMetaData.empty()
        
        % PaddingValue (scalar)
        PaddingValue
        
        % PaddingDirection (char)   Direction in which padding or
        %       truncation is applied. Either:
        %       'right'   : padding applied to the right of the sequence.
        %       All sequences begin at the same time.
        %       'left'    : padding applied to the left of the sequence.
        %       All sequences end at the same time.
        PaddingDirection

        % MinLength (int)   Minimum sequence length as specified by the
        % MinLength property of a sequence input layer. Mini-batches must
        % not have sequence length less than MinLength
        MinLength
    end
    
    properties(Access = private, Dependent)
        % ClassNames (cellstr) Array of class names corresponding to
        %            training data labels.
        ClassNames;
    end
    
    properties (Access = ?nnet.internal.cnn.dispatcher.mixin.DistributableDispatcher)
        % Data  (cell array)     A copy of the data in the workspace. This
        % is a numObservations-by-1 cell array, which for each observation
        % contains a DataSize-by-sequenceLength numeric array.
        Data
        
        % Response   A copy of the response data in the workspace. Either:
        % - numObservations-by-1 categorical vector 
        % - numObservations-by-1 cell array, which for each observation
        % contains a 1-by-sequenceLength categorical vector.
        Response
    end

    properties(Constant)
        DispatcherConstructor = @nnet.internal.cnn.dispatcher.SequenceClassificationDispatcher;
    end
    
    properties
        % Precision   Precision used for dispatched data
        Precision
        
        % EndOfEpoch    Strategy to choose how to cope with a number of
        % observation that is not divisible by the desired number of mini
        % batches
        %
        % Allowed values: 'truncateLast', 'discardLast'
        EndOfEpoch
    end
    
    methods
        function this = SequenceClassificationDispatcher(data, response, miniBatchSize, ...
                sequenceLength, endOfEpoch, paddingValue, paddingDirection, precision, networkInfo)
            % SequenceClassificationDispatcher   Constructor for sequence
            % classification dispatcher
            %
            % data              - cell array of sequences for training. The
            %                   number of elements in the cell array is the
            %                   number of training observations. Each
            %                   sequence has dimension D x S. D is fixed
            %                   for all observations, but S may vary per
            %                   observation
            % response          - Data responses in the form of either:
            %                   numObservations x 1  categorical array
            %                   numObservations x 1 cell array. 
            %                   The cell array must contain a 1 x S
            %                   categorical vector for each observation,
            %                   with S corresponding to the S of the data
            %                   at that observation
            % miniBatchSize     - Size of a mini batch expressed in number
            %                   of examples
            % sequenceLength    - Strategy to determine the length of the
            %                   sequences used per mini-batch. Options
            %                   are:
            %                   'shortest' to truncate all sequences in a
            %                   batch to the length of the shortest
            %                   sequence (default)
            %                   'longest' to pad all sequences in a
            %                   batch to the length of the longest sequence
            %                   Integer to pad or truncate all the
            %                   sequences in a batch to a specific integer
            %                   length.
            % endOfEpoch        - Strategy to choose how to cope with a
            %                   number of observations that is not
            %                   divisible by the desired number of mini
            %                   batches. One of: 
            %                   'truncateLast' to truncate the last mini
            %                   batch
            %                   'discardLast' to discard the last mini
            %                   batch (default)
            % paddingValue      - Scalar value used to pad sequences where
            %                   necessary. The default is 0.
            % paddingDirection  - Determine whether padding is applied to
            %                   before a sequence (left) or after a
            %                   sequence (right). One of:
            %                   'right' for right-padding
            %                   'left' for left-padding
            % precision         - What precision to use for the dispatched
            %                   data. Values are:
            %                   'single'
            %                   'double' (default).
            % networkInfo       - Instance of nnet.internal.cnn.NetworkInfo.
            
            % Assign data and response
            dataSize = networkInfo.InputSizes{1};
            responseSize = networkInfo.OutputSizes{1};
            minLength = networkInfo.MinSequenceLength;
            [dispatcherFormat, data, response] = iGetDispatcherFormat(data, response);
            this.Data = data;
            this.DataSize = dataSize;
            this.NumObservations = numel( data );
            [categories, response] = iGetCategoriesAndResponse(response, dispatcherFormat);
            this.Response = response;
            this.ResponseMetaData = nnet.internal.cnn.response.ClassificationMetaData( categories );
            this.ResponseSize = responseSize;
            this.DispatcherFormat = dispatcherFormat;

            if isnumeric(sequenceLength) && (sequenceLength < minLength)
                iThrowInvalidSeqLenError(dispatcherFormat, sequenceLength, minLength);
            end
            
            % Assign properties
            this.MinLength = minLength;
            this.SequenceLength = sequenceLength;
            this.EndOfEpoch = endOfEpoch;
            this.PaddingValue = paddingValue;
            this.PaddingDirection = paddingDirection;
            this.Precision = precision;
            this.MiniBatchSize = miniBatchSize;
            this.OrderedIndices = 1:this.NumObservations;
            
            % Assign read strategy
            readStrategy = iGetReadStrategy( dispatcherFormat, this.ClassNames, responseSize );
            
            % Assign next strategy
            this.NextStrategy = iGetNextStrategy(dispatcherFormat, ...
                readStrategy, ...
                sequenceLength, ...
                dataSize, ...
                responseSize, ...
                paddingValue, ...
                paddingDirection, ...
                minLength);
        end
        
        function names = get.ClassNames(this)
            names = categories(this.ResponseMetaData.Categories);
            names = cellstr(names);
        end
    end
end

function [dispatcherFormat, data, response] = iGetDispatcherFormat(data, response)
if isempty(response)
    % prediction only case
    dispatcherFormat = 'predict';
    % wrap data into cell if it is numeric (single observation)
    if isnumeric( data ) && ~isempty(data)
        data = { data };
    end
elseif iscell(data) && iscategorical(response)
    % seq2one format
    dispatcherFormat = 'seq2one';
elseif iscell(data) && iscell(response)
    % multi-observation seq2seq format
    dispatcherFormat = 'seq2seq';
elseif isnumeric(data) && iscategorical(response)
    % seq2seq with one observation format. Wrap data/response into cells
    dispatcherFormat = 'seq2seq';
    data = { data };
    response = { response };
end
end

function [cats, response] = iGetCategoriesAndResponse(response, dispatcherFormat)
switch dispatcherFormat
    case 'predict'
        % Empty response => prediction only
        cats = categorical();
    case 'seq2one'
        % Numeric response => seq2one problem
        cats = nnet.internal.cnn.util.categoriesFromResponse(response);
    case 'seq2seq'
        % Cell array of sequences => seq2seq problem
        allResponses = [ response{:} ];
        cats = nnet.internal.cnn.util.categoriesFromResponse(allResponses);        
        classNames = categories( cats );
        for ii = 1:numel( response )
            response{ii} = setcats( response{ii}, classNames );
        end
end
end

function strategy = iGetNextStrategy(dispatcherFormat, readStrategy, sequenceLength, dataSize, responseSize, paddingValue, paddingDirection, minLength)
switch dispatcherFormat
    case 'seq2seq'
        switch sequenceLength
            case 'longest'
                strategy = nnet.internal.cnn.sequence.Seq2SeqLongestStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
            case 'shortest'
                strategy = nnet.internal.cnn.sequence.Seq2SeqShortestStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
            otherwise
                strategy = nnet.internal.cnn.sequence.Seq2SeqFixedStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
        end
    case {'seq2one', 'predict'}
        switch sequenceLength
            case 'longest'
                strategy = nnet.internal.cnn.sequence.Seq2OneLongestStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
            case 'shortest'
                strategy = nnet.internal.cnn.sequence.Seq2OneShortestStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
            otherwise
                strategy = nnet.internal.cnn.sequence.Seq2OneFixedStrategy(readStrategy, dataSize, responseSize, paddingValue, paddingDirection, minLength);
        end
end
end

function strategy = iGetReadStrategy( dispatcherFormat, classNames, responseSize )
strategy.readDataFcn = iReadData();
strategy.readResponseFcn = iReadResponse( dispatcherFormat, classNames, responseSize );
end

function fcn = iReadData()
fcn = @(data, indices)data( indices );
end

function fcn = iReadResponse( dispatcherFormat, classNames, responseSize )
switch dispatcherFormat
    case 'predict'
        fcn = @(response, indices)[];
    case 'seq2one'
        fcn = @(response, indices)iDummify( response( indices ), classNames, responseSize );
    case 'seq2seq'
        fcn = @(response, index)iDummify( response{ index }, classNames, responseSize );
end
end

function dummy = iDummify(categoricalIn, classNames, responseSize)
% dummify   Dummify a categorical vector of size numObservations x 1 to
% return a matrix of size numClasses x numObservations
categoricalIn = reordercats( categoricalIn, classNames );
numObservations = numel(categoricalIn);
dummifiedSize = [responseSize(end), numObservations];
dummy = zeros(dummifiedSize);
categoricalIn = reshape( categoricalIn, 1, numel( categoricalIn ) );
idx = sub2ind(dummifiedSize, single(categoricalIn), 1:numObservations);
idx(isnan(idx)) = [];
dummy(idx) = 1;
dummy = reshape(dummy,[responseSize, numObservations]);
end

function iThrowInvalidSeqLenError(dispatcherFormat, seqLen, minLen)
if dispatcherFormat == "predict"
    error( message( 'nnet_cnn:internal:cnn:SequenceDispatcher:SeqLenLessThanMinLenInference', seqLen, minLen ) );
else
    error( message( 'nnet_cnn:internal:cnn:SequenceDispatcher:SeqLenLessThanMinLenTraining', seqLen, minLen ) );
end
end
