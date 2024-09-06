classdef seModel < handle
    %SEMODEL Speech enhancement for B-format Ambisonic Data

    properties (GetAccess=public,SetAccess=private)
        Learnables
        Parameters
    end

    methods
        function obj = seModel(netFolder)
            %SEMODEL Create a speech enhancer
            %   speechEnhancer = seModel(F) creates an object to perform
            %   FaSNet speech enhancement using B-format ambisonic data.
            %   Specify the location of the learnables of the FaSNet model
            %   as F.

            model = load(fullfile(netFolder,"FasNetParameters.mat"));
            obj.Learnables = model.Learnables;
            obj.Parameters = model.Parameters;
        end

        function y = enhanceSpeech(obj,x)
            %ENHANCESPEECH Enhance speech
            %   y = enhanceSpeech(speechEnhancer,x) passes the B-format
            %   ambisonic audio data in x through a FaSNet model. The
            %   FaSNet model applies end-to-end deep beamforming to the
            %   ambisonic data and returns a speech-enhanced mono signal.
            
            y = preprocessSignal(obj,x,obj.Parameters.AnalysisLength);
            y = dlarray(y);
            y = FaSNet(y,obj.Parameters,obj.Learnables);
            y = gather(extractdata(y));
            y = y(:);
            y = y(1:size(x,1));
        end
    end
    methods (Access=private)
        function y = preprocessSignal(~,x,L)
            %preprocessSignal Preprocess signal for FaSNet
            % y = preprocessSignal(x,L) splits the multi-channel
            % signal x into analysis frames of length L and hop L. The output is a
            % L-by-size(x,2)-by-numHop array, where the number of hops depends on the
            % input signal length and L.

            % Cast the input to single precision
            x = single(x);

            % Get the input dimensions
            N = size(x,1);
            nchan = size(x,2);

            % Pad as necessary.
            if N<L
                numToPad = L-N;
                x = cat(1,x,zeros(numToPad,size(x,2),like=x));
            else
                numHops = floor((N-L)/L) + 1;
                numSamplesUsed = L+(L*(numHops-1));
                if numSamplesUsed < N
                    numSamplesUnused = N-numSamplesUsed;
                    numToPad = L - numSamplesUnused;
                    x = cat(1,x,zeros(numToPad,nchan,like=x));
                end
            end

            % Buffer the input signal
            x = audio.internal.buffer(x,L,L);

            % Reshape the signal to Time-Channel-Hop.
            numHops = size(x,2)/nchan;
            x = reshape(x,L,numHops,nchan);
            y = permute(x,[1,3,2]);
        end
    end
end

function signalBF = FaSNet(X,parameters,learnables)

% Calculate derived parameters
contextWindow = 3*parameters.WindowLength;
filterDimension = 2*parameters.WindowLength+1;
numTotalChannels = size(X,2);

% Split input into chunks
[referenceContext,otherSegment,otherContext,rest] = ...
    segmentInputs(X,parameters.WindowLength,parameters.WindowLength);

[~,~,L,B] = size(referenceContext); % 3*win,1,L,B
% size(otherSegment) == [win,numTotalChannels-1,L,B];
% size(otherContext) == [contextWindow,nmic-1,L,B];

%% Step 1: Denoise Reference Mic

nmic = 1;

% Normalized Cross-Correlation
% Calculate cosine similarity between reference mic with context and the
% other segments. Then average over other segments.
refCosSim = ncc(referenceContext,otherSegment); % 2*win+1, nmic-1, L, B
refCosSim = mean(refCosSim,2); % 2*win+1, 1, L, B

% Temporal Convolution Network (TCN 1)
refFeature = reshape(referenceContext,contextWindow,B*L*1); % (2*context + win), B,L,1
refFeature = dlconv(refFeature, ...
    learnables.TCN.conv.weight, ...
    0, ...
    DataFormat="TBC", ...
    WeightsFormat="UCT");
refFeature = reshape(refFeature,L,B,[]);
refFeature = groupnorm(refFeature,1, ...
    learnables.TCN.norm.offset, ...
    learnables.TCN.norm.scaleFactor, ...
    DataFormat="TBC");

% Learn the beamforming filter to denoise reference mic.
refCosSim = permute(refCosSim,[1,2,4,3]);
refCosSim = reshape(refCosSim,filterDimension,nmic*B,L);
refFeature = permute(refFeature,[3,2,1]);
refFilter = learnBeamformer(refFeature,refCosSim,learnables.Beamformer1,parameters); % numHop-by-filterDim-by-(nmic*B)

% Reshape filter
refFilter = permute(refFilter,[1,3,2]);               % (nmic*B)-by-numHop-by-filterDim
refFilter = reshape(refFilter,[],1,filterDimension);  % (nmic*B*numHop)-by-1-by-filterDim
refFilter = permute(refFilter,[3,2,4,1]);             % filterDim-by-1-by-1-by-(nmic*B*numHop)

% Apply beamforming filter using channel-wise separable convolution.
referenceContext = permute(referenceContext,[1,2,3,4]);               % contextDim-by-nmic-by-numHop-by-B
referenceContext = reshape(referenceContext,contextWindow,nmic*B*L);  % contextDim-by-(nmic*B*numHop)

refBF = dlconv(referenceContext,refFilter,0,DataFormat="SC");
refBF = reshape(refBF,parameters.WindowLength,nmic,L,B);

refBF = refBF + 1e-8; % Offset to keep gradient descent stable

%% Step 2: Creating the beam-formed signal

nmic = numTotalChannels-1;

% Normalized Cross-Correlation
% Calculate cosine similarity between other mics with context and the
% reference segment.
otherCosSim = ncc(otherContext,refBF); % filterDimension-by-nmic-by-numHops-by-B

% Temporal Convolution Network (TCN 2)
otherFeature = permute(otherContext,[1,3,2,4]);
otherFeature = reshape(otherFeature,contextWindow,B*L*nmic); % (2*context + win),B,L,1
otherFeature = dlconv(otherFeature, ...
    learnables.TCN.conv.weight, ...
    0, ...
    DataFormat="TBC", ...
    WeightsFormat="UCT");
otherFeature = reshape(otherFeature,L,B*nmic,[]);
otherFeature = groupnorm(otherFeature,1, ...
    learnables.TCN.norm.offset, ...
    learnables.TCN.norm.scaleFactor, ...
    DataFormat="TBC");
% size(otherFeature) == encoderDimension-by-(nmic*B)-by-numHops

% Learn the beamforming filter to denoise other mics.
otherCosSim = permute(otherCosSim,[1,2,4,3]);
otherCosSim = reshape(otherCosSim,filterDimension,nmic*B,L);
otherFeature = permute(otherFeature,[3,2,1]);
otherFilter = learnBeamformer(otherFeature,otherCosSim,learnables.Beamformer2,parameters);
% size(otherFilter) == numHop-by-filterDim-by-(nmic*B)

% Reshape filter
otherFilter = permute(otherFilter,[1,3,2]);               % (nmic*B)-by-numHop-by-filterDim
otherFilter = reshape(otherFilter,[],1,filterDimension);  % (nmic*B*numHop)-by-1-by-filterDim
otherFilter = permute(otherFilter,[3,2,4,1]);             % filterDim-by-1-by-1-by-(nmic*B*numHop)

% Channel-wise separable (aka depth-wise separable) convolution.
otherContext = permute(otherContext,[1,3,2,4]); % (2*context + win), L, nmic-1, B
otherContext = reshape(otherContext,contextWindow,L*nmic*B);

otherBF = dlconv(otherContext,otherFilter,0,DataFormat="SC");
otherBF = reshape(otherBF,[],L,nmic,B);

%% Step 3: Reconstruct Signal

% Combine the beamformed reference channel and other channels
otherBF = permute(otherBF,[1,3,2,4]);
allBF = cat(2,refBF,otherBF);

% Reconstruct signal using overlap add
signalBF = allBF((parameters.WindowLength/2)+1:parameters.WindowLength,:,1:end-1,:) + allBF(1:(parameters.WindowLength/2),:,2:end,:);

signalBF = permute(signalBF,[1,3,2,4]);
signalBF = reshape(signalBF,[],numTotalChannels,B);

% Average across channels
signalBF = mean(signalBF(1:end-rest,:,:),2);
signalBF = permute(signalBF,[1,3,2]);

end

function [referenceContext,otherSegment,otherContext,rest] = segmentInputs(W,K,context)
% Segment signal into chunks with specified context.

P = K/2; % Stride/hop length == chunk length / 2.

% Pad input so its divisible by hop size P.
[W,rest] = iPadInput(W,K);

numChan = size(W,2);
numBatch = size(W,3);

% Buffer the input
allSegment = permute(iBuffer(W,K,P),[4,3,2,1]); % Batch,Channel,Window,Hop

% Pad according to context size
padContext = zeros(context,numChan,numBatch,like=W);
W = cat(1,padContext,W,padContext);

% Buffer the padded input
allContext = permute(iBuffer(W,2*context+K,P),[4,3,2,1]); % channel-time step-window

referenceContext = allContext(:,1,:,:);
referenceContext = permute(referenceContext,[4,2,3,1]);

otherSegment = allSegment(:,2:end,:,:);
otherSegment = permute(otherSegment,[4,2,3,1]);

otherContext = allContext(:,2:end,:,:);
otherContext = permute(otherContext,[4,2,3,1]);

    function Y = iBuffer(W,K,P)
        numHops = floor((size(W,1)-K)/P) + 1;
        idx1 = repmat((1:K)',1,numHops) + (0:numHops-1)*P;
        idx2 = idx1 + reshape((0:size(W,2)-1)*size(W,1),1,1,[]);

        idx3 = idx2 + reshape((0:size(W,3)-1)*size(W,1)*size(W,2),1,1,1,[]);

        Y = reshape(W(idx3(:)),K,numHops,[],size(W,3));
    end
    function [x,rest] = iPadInput(x,win)
        [nsamples,nmic,nbatch] = size(x);
        stride = win/2;

        % Pad signal at end to match the window/stride size.
        rest = win - mod((stride + mod(nsamples,win)),win);

        if rest > 0
            pad = zeros(rest,nmic,nbatch,like=x);
            x = cat(1,x,pad);
        end
        padAux = zeros(stride,nmic,nbatch,like=x);
        x = cat(1,padAux,x,padAux);

        rest = dlarray(rest);
    end
end

function [Y,rest] = splitFeature(X,ss)

hop = floor(ss/2);

[Xpadded,rest] = iPadInput(X,ss);

Y = iBuffer3D(Xpadded,ss,hop);

    function y = iBuffer3D(x,windowLength,hopLength)
        % Apply a buffer to a 3d signal along the third dimension. The buffered
        % signals are placed along the 4th dimension.

        numHops = floor(size(x,3)/hopLength - 1);
        pageIndex = repmat((1:windowLength)',1,numHops)+((0:numHops-1)*hopLength);

        y = [];
        for kk = 1:size(x,4)
            yy = x(:,:,pageIndex(:),kk);
            yy = reshape(yy,size(x,1),size(x,2),windowLength,numHops);
            y = cat(5,y,yy);
        end
    end
    function [x,rest] = iPadInput(x,win)
        [dim1,dim2,dim3,dim4] = size(x);
        stride = ceil(win/2);

        % Pad signal at end to match the window/stride size.
        rest = win - mod((stride + mod(dim3,win)),win);

        if rest > 0
            pad = zeros(dim1,dim2,rest,dim4,like=x);
            x = cat(3,x,pad);
        end
        padAux = zeros(dim1,dim2,stride,dim4,like=x);
        x = cat(3,padAux,x,padAux);

        rest = dlarray(rest);
    end
end
function score = ncc(ref,target)
% The paper calls this normalized cross correlation, but really its cosine
% similarity.

nmic = max(size(ref,2),size(target,2));
nhop = size(ref,3);
nbatch = size(ref,4);
L = size(target,1);

if size(target,2) > size(ref,2)
    ref = repmat(ref,[1,nmic,1,1]);
elseif size(target,2) < size(ref,2)
    target = repmat(target,[1,nmic,1,1]);
end

% Collapse batch and sequence dimensions.
ref = reshape(ref,size(ref,1),size(ref,2),[]);
target = reshape(target,size(target,1),size(target,2),[]);

Ar = reshape(ref,size(ref,1),[]);
weights = ones(L,1,1,size(Ar,2));
referenceNorm = dlconv(Ar.^2,weights,0,DataFormat="SC");
referenceNorm = sqrt(referenceNorm) + 1e-8;

targetNorm = sqrt(sum(target.^2,1));
targetNorm = reshape(targetNorm,1,[]) + 1e-8;

target = reshape(target,size(target,1),1,1,[]);
ref = reshape(ref,size(ref,1),[]);
cosineSimilarity = dlconv(ref,target,0,DataFormat="SC");

cosineSimilarity = cosineSimilarity./(referenceNorm.*targetNorm);

score = reshape(cosineSimilarity,size(cosineSimilarity,1),nmic,nhop,nbatch);
end

function Y = learnBeamformer(a,b,learnables,parameters)

input = cat(1,a,b);

% Bottleneck
encoderFeature = dlconv(input, ...
    learnables.BN.conv.weight, ...
    0, ...
    DataFormat="CBT", ...
    WeightsFormat="UCT");

% Split the encoder output into overlapped segments
[encoderFeatureSegmented,encoderRest] = splitFeature(encoderFeature,parameters.SegmentSize);

out = encoderFeatureSegmented;
for ii = 1:parameters.NumDPRNNBlocks
    out = DPRNN(out,learnables.("DPRNN_"+ii));
end
out = permute(out,[2,1,3,4]);

% Output layer (prelu + conv2d)
act = max(out,0) + learnables.Output.prelu.alpha .* min(0,out);

output = dlconv(act, ...
    learnables.Output.conv.weight, ...
    learnables.Output.conv.bias, ...
    DataFormat="BCSS", ...
    WeightsFormat="UC");
output = permute(output,[3,4,2,1]);

% Overlap-and-add of the outputs
X = iMergeFeature(output,parameters.SegmentSize,encoderRest);

% Gated output layer for filter generation
X = permute(X,[3,2,1]);

X1 = dlconv(X, ...
    learnables.GenerateFilter.X1.weight, ...
    learnables.GenerateFilter.X1.bias, ...
    DataFormat="SCB", ...
    WeightsFormat="UC");
X1 = tanh(X1);

X2 = dlconv(X, ...
    learnables.GenerateFilter.X2.weight, ...
    learnables.GenerateFilter.X2.bias, ...
    DataFormat="SCB", ...
    WeightsFormat="UC");
X2 = sigmoid(X2);

Y = X1.*X2;

    function output = iMergeFeature(x,ss,rest)

        A = permute(x,[3,4,1,2,5]);
        windowLength = size(A,3);
        hopLength = ss/2;

        % Overlap add
        output = A(:,:,hopLength+1:windowLength,1:end-1,:,:) + A(:,:,1:hopLength,2:end,:);

        output = reshape(output,size(A,1),size(A,2),[],size(A,5));
        output = output(:,:,1:end-rest,:);
        output = permute(output,[2,1,3,4]);

    end
end
function Y = DPRNN(path1Input,learnables)

% First path
input = permute(path1Input,[4,2,3,1]);
[dim1,batchSize,dim2,~] = size(input);

hiddenDimension = size(learnables.pass1.rnn.forward.recurrentWeights,2);

input = reshape(input,dim2*batchSize,dim1,[]);

H0 = dlarray(zeros(hiddenDimension,dim2*batchSize,like=path1Input),"CB");
C0 = dlarray(zeros(hiddenDimension,dim2*batchSize,like=path1Input),"CB");

% Intra-chunk RNN
outputRowForward = lstm(input,H0,C0, ...
    learnables.pass1.rnn.forward.weights, ...
    learnables.pass1.rnn.forward.recurrentWeights, ...
    learnables.pass1.rnn.forward.bias, ...
    DataFormat="BTC");
outputRowReverse = lstm(flip(input,2),H0,C0, ...
    learnables.pass1.rnn.reverse.weights, ...
    learnables.pass1.rnn.reverse.recurrentWeights, ...
    learnables.pass1.rnn.reverse.bias, ...
    DataFormat="BTC");
outputRow = cat(3,outputRowForward,flip(outputRowReverse,2));

% Projection
outputRow = permute(outputRow,[3,2,1]);
outputRow = reshape(outputRow,size(outputRow,1),[]);
outputRow = fullyconnect(outputRow, ...
    learnables.pass1.projection.weights, ...
    learnables.pass1.projection.bias, ...
    DataFormat="CB");
outputRow = reshape(outputRow,[],dim1,dim2*batchSize);

% Norm
outputRow = reshape(outputRow,size(outputRow,1),dim2,dim1,[]);
outputRow = groupnorm(outputRow,1, ...
    learnables.pass1.norm.offset, ...
    learnables.pass1.norm.scaleFactor, ...
    DataFormat="CUSB");
outputRow = permute(outputRow,[1,4,2,3]);

% Sum
path1Output = outputRow + path1Input;

% Second path
path2Input = permute(path1Output,[3,2,4,1]);
path2Input = reshape(path2Input,dim1*batchSize,dim2,[]);
path2Input = permute(path2Input,[3,1,2]);

H0 = dlarray(zeros(hiddenDimension,dim1*batchSize,like=path1Input),"CB");
C0 = dlarray(zeros(hiddenDimension,dim1*batchSize,like=path1Input),"CB");

% Inter-chunk RNN
outputColForward = lstm(path2Input,H0,C0, ...
    learnables.pass2.rnn.weights, ...
    learnables.pass2.rnn.recurrentWeights, ...
    learnables.pass2.rnn.bias, ...
    DataFormat="CBT");
outputColReverse = lstm(flip(path2Input,3),H0,C0, ...
    learnables.pass2.rnn.reverse.weights, ...
    learnables.pass2.rnn.reverse.recurrentWeights, ...
    learnables.pass2.rnn.reverse.bias, ...
    DataFormat="CBT");
outputCol = cat(1,outputColForward,flip(outputColReverse,3));

% Projection
outputCol = permute(outputCol,[1,3,2]);
outputCol = reshape(outputCol,size(outputCol,1),[]);
outputCol = fullyconnect(outputCol, ...
    learnables.pass2.projection.weights, ...
    learnables.pass2.projection.bias, ...
    DataFormat="CB");
outputCol = reshape(outputCol,[],dim1,dim2,batchSize);

% Norm
outputCol = groupnorm(outputCol,1, ...
    learnables.pass2.norm.offset, ...
    learnables.pass2.norm.scaleFactor, ...
    DataFormat="CSUB");
outputCol = permute(outputCol,[1,4,3,2]);

% Sum
Y = outputCol + path1Output;

end