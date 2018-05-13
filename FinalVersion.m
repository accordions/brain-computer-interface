% Final Project - Motor Contest

% This code contains 3 parts:
% 1. Algorithm for downloading IEEG data and extracting features
% (Raw IEEG data is never saved; only the extracted features are saved.)
% 2. Algorithm for downloading glove data
% 3. Algorithm for cross-validation
% 4. Algorithm for generating testing predictions

%% ============ Part I: IEEG DATA DOWNLOADING AND FEATURE EXTRACTION ==============

%% Initialization
clear; clc;
subjectNum = 3;
dataType = 3; % 1 for training IEEG, 2 for training glove, 3 for testing IEEG
session = IEEGSession(['I521_A' num2str(subjectNum+8,'%.4d') '_D00' num2str(dataType)], ...
    'jayyu0528', '\jay_ieeglogin.bin');

%% Parameter Setting
numFeats = 3; % <<<======== Change manually when new features are added
numChannels = length(session.data.channels);
sampRate = session.data.sampleRate; % Hz
if dataType ~= 3
    timePerFile = 400; % sec
else
    timePerFile = 200;
end
sampPerFile = sampRate * timePerFile;
numFingers = 5;

epochTime = 0.1; % sec
epochShift = 0.05; % sec
epochSize = epochTime * sampRate;
epochStep = epochShift * sampRate;

% Matrix Initialization
epochPerFile = (sampPerFile / epochStep - 1)
% numFeats = length(featFns);
numEpochs = epochPerFile;
featVec = zeros(numChannels,numEpochs,numFeats);
epochStart = zeros(1,numEpochs);
epochEnd = zeros(1,numEpochs);

%% Feature Extraction
% NFFT = max(256,2^nextpow2(numEpochs)); 
NFFT = 128*8; % <=== Change FFT resolution (must be power of 2);
f1 = [8 60 100]; % lower bound
f2 = [13 100 200]; % upper bound
for ch = 1:numChannels
    ch
    data = session.data.getvalues(1:sampPerFile,ch);
    % FFT Features
    [S,F,T] = spectrogram(data,epochSize,epochStep,NFFT,sampRate);
    Sabs = abs(S);
    for f = 1:length(f1)
    % FFT Features
        featIndex = f; % Change if more features before FFT ones
        FOI = find((F > f1(f))&(F < f2(f)));
        featVec(ch,:,featIndex) = mean(Sabs(FOI,:),1);
    end
end

%% Save Results to Current Directory
switch dataType
    case 1
        save(['Data_S' num2str(subjectNum) '_Train']);
    case 3
        save(['Data_S' num2str(subjectNum) '_Test']);
end

%% ================== END OF Part I: IEEG DATA PRE-PROCESSING =================
return

%% ============ Part II: Glove Data Downloading and Processing =========
% Warning: Algorithm will save to current directory
load('Data_S1_Train','sampRate','sampPerFile','epochShift','numFingers');
for subjectNum = 1:3
    session = IEEGSession(['I521_A' num2str(subjectNum+8,'%.4d') '_D00' num2str(2)], ...
        'jayyu0528', '\jay_ieeglogin.bin');
    gloveData = session.data.getvalues(1:sampPerFile,1:numFingers);
    sampPerDownSamp = epochShift * sampRate;
    numDownSamps = sampPerFile / sampPerDownSamp;
    gloveDataDS = zeros(numDownSamps,numFingers);
    for f = 1:numFingers
        gloveDataDS(:,f) = decimate(gloveData(:,f),sampPerDownSamp);
    end
    save(['Data_S' num2str(subjectNum) '_Glove']);
end


%% ================== Part III: CROSS-VALIDATION ========================
% Execute this entire block for cross-validation.
% Set current directory to where the data files for ALL 3 patients are
% saved. CV runs through all three patients to generate correlation.

clear; clc;
% Set Parameters
numSubjects = 3;
numFolds = 4;
tempTrainBlock = {[1,2],[2;3],[3,4]};
numRounds = length(tempTrainBlock);

tempTrainCorr = zeros(numSubjects,numRounds);
tempTestCorr = zeros(numSubjects,numRounds);

% smoothing window
epochTime = 0.1; % sec
epochShift = 0.05; % sec
sampRate = 1000; % Hz
epochSize = epochTime * sampRate;
epochStep = epochShift * sampRate;

for subjectNum = 1:3
    subjectNum
    for rNum = 1:length(tempTrainBlock)
        rNum
        N = 4;
        incSelf = 1; % 0 = No; 1 = Yes
        startOffset = 50;
        switch incSelf
            case 0
                trimFn = @(x,N,dim) trimExcSelf(x,N,dim);
            case 1
                trimFn = @(x,N,dim) trimIncSelf(x,N,dim);
        end
        
        % =============== Load Data Files =================
        load(['Data_S' num2str(subjectNum) '_Train'],'featVec','epochPerFile',...
            'epochStep','epochSize','sampPerFile');
        trainFeatVec = featVec;
        
        % feature normalization
        for i = 1:size(trainFeatVec,1) % subject number
            for j = 1:size(trainFeatVec,3) % features
        meanFeat = mean(trainFeatVec(i,:,j));
        stdFeat = std(trainFeatVec(i,:,j));
        trainFeatVec(i,:,j) = (trainFeatVec(i,:,j)-meanFeat)/stdFeat;
            end
        end
        % end feature normalization
        
        
        trainNumEpochs = epochPerFile;
        trainSampPerFile = sampPerFile;
        load(['Data_S' num2str(subjectNum) '_Glove'],'gloveData','gloveDataDS');
        clear featVec epochPerFile sampPerFile;
        
        % =============== Data Reshaping ==================
        numFeats = size(trainFeatVec,3);
        numChannels = size(trainFeatVec,1);
        numFeats2 = numChannels * numFeats;
        % Reshape Rule: Channel Major Order - Ch1F1, Ch1F2, ... Ch2F1, Ch2F2, ...
        % Training FeatVec Reshaping
        trainFeatVecR = zeros(trainNumEpochs,numFeats2);
        for ch = 1:size(trainFeatVec,1)
            trainFeatVecR(:,((ch-1)*numFeats+1:ch*numFeats)) = trainFeatVec(ch,:,:);
        end
        
        % ============== Epoch Sorting ==============
        tempTestBlock = [1,2,3,4];
        tempTestBlock(tempTrainBlock{rNum}) = [];
        epochInFold = cell(1,numFolds);
        epochPerFold = ceil(trainNumEpochs/4);
        for f = 1:numFolds
            epochInFold{f} = ...
                (1:(epochPerFold-1))+(f-1)*epochPerFold;
        end
        tempTrainEpochs = epochInFold{tempTrainBlock{rNum}(1)}(1):...
            epochInFold{tempTrainBlock{rNum}(2)}(end);
        tempTestEpochs1 = epochInFold{tempTestBlock(1)};
        tempTestEpochs2 = epochInFold{tempTestBlock(2)};
        tempEpochs = {tempTrainEpochs,tempTestEpochs1,tempTestEpochs2};
        
        % =========  Data Setup for Filter Calculation =======
        % Glove Data Pre-Processing
        tempM = [size(tempTrainEpochs,2),size(tempTestEpochs1,2),...
            size(tempTestEpochs2,2)] - N + 1;
        tempGloveDataDS = gloveDataDS([tempEpochs{1} tempEpochs{1}(end)+1],:);
        % x = 0.14;
        % tempGloveDataDS(find(tempGloveDataDS < x)) = x;
        tempGloveDataDSR = trimFn(tempGloveDataDS,N,1);
        tempR = cell(1,3);
        % Order: Training, Testing1, Testing2
        for fd = 1:length(tempM)
            % R Matrix Setup
            tempR{fd} = zeros(tempM(fd),1+numFeats2*N);
            tempR{fd}(:,1) = 1;
            tempFeatVecR = trainFeatVecR(tempEpochs{fd},:);
            for t = 1:tempM(fd)
                for f = 1:numFeats2
                    RindexStart = (f-1)*N+2;
                    RindexEnd = RindexStart+N-1;
                    tempR{fd}(t,RindexStart:RindexEnd) = tempFeatVecR(t:(t+N-1),f)';
                end
            end
        end
        
        % ============  Filter Calculation =============
        % Train Filter From Training Data
        Filter = mldivide(tempR{1}'*tempR{1},tempR{1}'*tempGloveDataDSR);
        
        % =========== Generate Predictions ============
        tempPred = cell(1,3);
        for r = 1:3
            tempPred{r} = tempR{r}*Filter;
        end
        
        % ============= Spline Interpolation ============
        
        trainGlobalIndex = (startOffset:epochStep:trainSampPerFile);
        tempGlobalIndex = cell(1,3);
        tempGlovalIndexR = cell(1,3);
        meanCorr = zeros(1,3);
        
        for r = 1:3
            tempGlobalIndex = ...
                trainGlobalIndex([tempEpochs{r} tempEpochs{r}(end)+1]);
            tempGlobalIndexR = trimFn(tempGlobalIndex,N,2);
            globalStartUS = tempGlobalIndex(1) - startOffset + 1;
            globalEndUS = tempGlobalIndex(end) - startOffset + epochStep;
            tempGloveData = gloveData(globalStartUS:globalEndUS,:);
            tempPredUS = zeros(length(globalStartUS:globalEndUS),5);
            tempPredUS1 = zeros(size(tempPredUS));
            tempPredUS2 = zeros(size(tempPredUS));
              
            for f = 1:5
                fstGlobalIndex = tempGlobalIndexR(1);
                lstGlobalIndex = tempGlobalIndexR(end);
                overWrite = (fstGlobalIndex:lstGlobalIndex);
                tempPredUS(overWrite-(globalStartUS-1),f) = ...
                    spline(tempGlobalIndexR,tempPred{r}(:,f),...
                    overWrite);
               % x = 0.14;
               % tempPredUS(find(tempPredUS(:,f)<x)) = x;
                
                %tempPredUS(:,f) = conv(tempPredUS(:,f),ones(1500,1)/1000,'same');
                tempPredUS1(:,f) = conv(tempPredUS(:,f),ones(1500,1)/1000,'same');
                tempPredUS2(:,f) = conv(tempPredUS1(:,f),ones(1500,1)/1000,'same');
                % tempPredUS2(:,f) = conv(tempPredUS2(:,f),ones(1500,1)/1000,'same');
               
                
                % upsampling for each finger <======================== NEW
                %{
                glove_data = tempGloveData(:,f);
                gloveMean = mean(glove_data);
                baseline = mean(glove_data(find(glove_data < gloveMean)));
                thresh = mean(glove_data(:)) + 0.9*std(glove_data(:));
                ind = tempPredUS(:,f) > thresh;
                ind2 = tempPredUS(:,f) < thresh;
                %}
                %tempPredUS(ind,f) = 5;
                %tempPredUS(ind2,f) = baseline;
                %tempPredUS(:,f)=smooth(tempPredUS(:,f),epochSize/10,'sgolay');
                % tempPredUS(:,f) = conv(tempPredUS(:,f),ones(1500,1)/1000,'same');    
            end
            % Generate Correlation
            tempCorr = corr(tempGloveData,tempPredUS2);
            diagnalCorr = 0;
            for i = 1:size(tempCorr,1)
                diagnalCorr = diagnalCorr + tempCorr(i,i);
            end
            meanTrainCorr = diagnalCorr / 5;
            meanCorr(r) = meanTrainCorr;
        end
        tempTrainCorr(subjectNum,rNum) = meanCorr(1);
        tempTestCorr(subjectNum,rNum) = mean(meanCorr(2:3));
    end
end

subjectCorr = [mean(tempTrainCorr,2),mean(tempTestCorr,2)]
CVCorr = mean(subjectCorr,1)

y2 = tempGloveData(1:100000,3);
mean(diag(corr(tempPredUS2(1:100000,3),y2)))
%{
storage = [];
for z = 1:16
tempPredUSnew = circshift(tempPredUS,[-z*50 0]);
tempPredUSnew (1:z*50,:) = 0;
y = tempPredUSnew(1:100000,3);
y2 = tempGloveData(1:100000,3);
storage(z) = mean(diag(corr(y,y2)));
end

%}
% visualize glove data & prediction

figure(2)
t = 1:100000;
y = tempPredUS2(1:100000,3);
y2 = tempGloveData(1:100000,3);
y3 = tempPredUS(1:100000,3);
y4 = tempPredUS1(1:100000,3);
mean(diag(corr(y,y2)))
mean(diag(corr(y4,y2)))
plot(t,y,'b',t,y4,'k',t,y3,'g',t,y2,'r')
legend ('convolution 2','convolution 1','original predictions','glovedata')
title('Subject 3, 3rd Finger: Actual vs Original & Post Processed Predictions')
xlabel('Time(ms)')
ylabel('Finger Amplitude')
%% ============== END OF Part III CROSS-VALIDATION ==================

%% ============ Part IV: GENERATE TEST PREDICTIONS ====================
clear; clc;
N = 3;
incSelf = 0; % 0 = No; 1 = Yes
startOffset = 1; % any value from 1 to 50 (epochStep)
switch incSelf
    case 0
        trimFn = @(x,N,dim) trimExcSelf(x,N,dim);
    case 1
        trimFn = @(x,N,dim) trimIncSelf(x,N,dim);
end
saveName = 'GhostPred0506_2';
for subjectNum = 1:3;
    % =============== Load Data Files =================
    load(['Data_S' num2str(subjectNum) '_Train'],'featVec','epochPerFile',...
        'epochStep');
    trainFeatVec = featVec;
    trainNumEpochs = epochPerFile;
    load(['Data_S' num2str(subjectNum) '_Test'],'featVec','epochPerFile');
    testFeatVec = featVec;
    testNumEpochs = epochPerFile;
    load(['Data_S' num2str(subjectNum) '_Glove'],'gloveData','gloveDataDS');
    clear featVec epochPerFile;
    
    % =============== Data Reshaping ==================
    numFeats = size(trainFeatVec,3);
    numChannels = size(trainFeatVec,1);
    numFeats2 = numChannels * numFeats;
    % Reshape Rule: Channel Major Order - Ch1F1, Ch1F2, ... Ch2F1, Ch2F2, ...
    % Training FeatVec Reshaping
    trainFeatVecR = zeros(trainNumEpochs,numFeats2);
    for ch = 1:size(trainFeatVec,1)
        trainFeatVecR(:,((ch-1)*numFeats+1:ch*numFeats)) = trainFeatVec(ch,:,:);
    end
    % Testing FeatVec Reshaping
    testFeatVecR = zeros(testNumEpochs,numFeats2);
    for ch = 1:size(testFeatVec,1)
        testFeatVecR(:,((ch-1)*numFeats+1:ch*numFeats)) = testFeatVec(ch,:,:);
    end
    
    % =========  Data Setup for Filter Calculation =======
    % Glove Data Pre-Processing
    trainM = trainNumEpochs - N + 1;
    testM = testNumEpochs - N + 1;
    gloveDataDSR = trimFn(gloveDataDS,N,1);
    
    % Training R Matrix Setup
    trainR = zeros(trainM,1+numFeats2*N);
    trainR(:,1) = 1;
    for t = 1:trainM
        for f = 1:numFeats2
            RindexStart = (f-1)*N+2;
            RindexEnd = RindexStart+N-1;
            trainR(t,RindexStart:RindexEnd) = trainFeatVecR(t:(t+N-1),f)';
        end
    end
    
    % Testing R Matrix Setup
    testR = zeros(testM,1+numFeats2*N);
    testR(:,1) = 1;
    for t = 1:testM
        for f = 1:numFeats2
            RindexStart = (f-1)*N+2;
            RindexEnd = RindexStart+N-1;
            testR(t,RindexStart:RindexEnd) = testFeatVecR(t:(t+N-1),f)';
        end
    end
    
    % ============  Filter Calculation =============
    % Train Filter From Training Data
    Filter = mldivide(trainR'*trainR,trainR'*gloveDataDSR);
    
    % =========== Generate Predictions ============
    % Training Predictions
    trainPred = trainR*Filter;
    % Testing Predictions
    testPred = testR*Filter;
    
    % ============= Spline Interpolation ============
    % Zero Padding is implicitly done by writing over only specific regions
    % on the PredUS matrix, initialized all by 0.
    % US represents Up-Sampled.
    
    % Training Prediction Up Sampling
    trainSampPerFile = (trainNumEpochs+1)*epochStep;
    trainIndexUS = (startOffset:epochStep:trainSampPerFile)';
    trainIndexUSR = trimFn(trainIndexUS,N,1); %
    trainPredUS = zeros(trainSampPerFile,5);
    for f = 1:5
        trainPredUS(trainIndexUSR(1):trainIndexUSR(end),f) = ...
            spline(trainIndexUSR,trainPred(:,f),...
            (trainIndexUSR(1):trainIndexUSR(end))');
        trainPredUS(:,f) = conv(trainPredUS(:,f),ones(1500,1)/1000,'same');
        trainPredUS(:,f) = conv(trainPredUS(:,f),ones(1500,1)/1000,'same'); 
    end
    
    % Testing Prediction Up Sampling
    testSampPerFile = (testNumEpochs+1)*epochStep;
    testIndexUS = (startOffset:epochStep:testSampPerFile)';
    testIndexUSR = trimFn(testIndexUS,N,1); %
    testPredUS = zeros(testSampPerFile,5);
    for f = 1:5
        testPredUS(testIndexUSR(1):testIndexUSR(end),f) = ...
            spline(testIndexUSR,testPred(:,f),(testIndexUSR(1):testIndexUSR(end))');
        testPredUS(:,f) = conv(testPredUS(:,f),ones(1500,1)/1000,'same');
        testPredUS(:,f) = conv(testPredUS(:,f),ones(1500,1)/1000,'same'); 
    end
    
    % ============ Training Correlation ================
    trainCorr = corr(gloveData,trainPredUS);
    diagnalCorr = 0;
    for i = 1:size(trainCorr,1)
        diagnalCorr = diagnalCorr + trainCorr(i,i);
    end
    meanTrainCorr = diagnalCorr / 5;
    disp(['Training Correlation for Subject #' num2str(subjectNum) ' is ' ...
        num2str(meanTrainCorr) '.']);
    eval(['sub' num2str(subjectNum) 'test_dg = testPredUS;']);
    
end
% Save File
save(saveName,'sub1test_dg','sub2test_dg','sub3test_dg');

