%% Massey Hand gesture classification 
clear all;
close all;

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
tic
inputSize = 28 * 28;
numClasses = 5;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
hiddenSizeL3 = 200;    % Layer 3 Hidden Size
hiddenSizeL4 = 200;    % Layer 4 Hidden Size
hiddenSizeL5 = 200;    % Layer 5 Hidden Size

sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
% lambda = 3e-3;         % weight decay parameter   
lambda = 3e-5;  
% Before Finetuning Test Accuracy: 67.409%
% After Finetuning Test Accuracy: 83.364%

beta = 3;              % weight of sparsity penalty term     
maxIter = 1000;

no_train = 600 ; 
total_samp = 1000;
no_test = 400;
%%======================================================================
%% STEP 1: Load data from the Massey database
IMG=0;

if IMG
% directory_train_1 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\1\';
% directory_train_2 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\2\';
% directory_train_3 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\3\';
% directory_train_4 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\4\';
% directory_train_5 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\5\';
% directory_train_6 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\6\';
% directory_train_7 = 'F:\Varun IRO2012013\gesture_autoencoder\hand\7\';
% directory_train_unlabelled = 'F:\Varun IRO2012013\gesture_autoencoder\hand\unlabelled\';


directory_train_1 = 'F:\varun\data\1\';
directory_train_2 = 'F:\varun\data\2\';
directory_train_3 = 'F:\varun\data\3\';
directory_train_4 = 'F:\varun\data\4\';
directory_train_5 = 'F:\varun\data\5\';
%directory_train_6 = 'D:\varun\gesture_autoencoder\hand\6\';
%directory_train_7 = 'D:\varun\gesture_autoencoder\hand\7\';
directory_train_unlabelled = 'F:\varun\data\unlabelled\';


Data1 = readImages(directory_train_1);
Data2 = readImages(directory_train_2);
Data3 = readImages(directory_train_3);
Data4 = readImages(directory_train_4);
Data5 = readImages(directory_train_5);
%Data6 = readImages(directory_train_6);
%Data7 = readImages(directory_train_7);

Data_unlabelled = readImages(directory_train_unlabelled);



trainData = [ Data1(:,1:no_train) , Data2(:,1:no_train) , Data3(:,1:no_train) ,...
              Data4(:,1:no_train) , Data5(:,1:no_train)];

testData = [ Data1(:,no_train+1:total_samp) , Data2(:,no_train+1:total_samp) , Data3(:,no_train+1:total_samp) ,...
              Data4(:,no_train+1:total_samp) , Data5(:,no_train+1:total_samp)];

          
trainLabels =  [ones(no_train,1) ; 2*ones(no_train,1); 3*ones(no_train,1) ;...
                4*ones(no_train,1); 5*ones(no_train,1)] ;

testLabels = [ones(no_test,1) ; 2*ones(no_test,1); 3*ones(no_test,1) ;...
                4*ones(no_test,1); 5*ones(no_test,1)] ;
            

AE_train_data =Data_unlabelled;

avg = mean([trainData,testData,AE_train_data],2);
sd = max([trainData,testData,AE_train_data],[],2)-min([trainData,testData,AE_train_data],[],2);



else
load('D:\Thesis\Autoencoder5layers\trainTestUnlabelledMeanSd.mat');
%load('F:\varun\data\unlabelledExtData.mat');
%73.550%
end

% Shuffle the data
unlabelled_size = size(AE_train_data,2);
ind_train = randperm(length(trainLabels));
ind_test = randperm(length(testLabels));
% ind_AE = randperm(96456);
ind_AE = randperm(unlabelled_size);
 
trainData = trainData(:,ind_train);
testData = testData(:,ind_test);
trainLabels = trainLabels(ind_train);
testLabels = testLabels(ind_test);
AE_train_data =  AE_train_data(:,ind_AE);


% normalize the data
% trainData = bsxfun(@minus,trainData,mean(trainData,2));
% trainData = bsxfun(@rdivide,trainData,(max(trainData,[],2)-min(trainData,[],2)));
% 
% testData = bsxfun(@minus,testData,mean(testData,2));
% testData = bsxfun(@rdivide,testData,(max(testData,[],2)-min(testData,[],2)));
%  
% AE_train_data = bsxfun(@minus,AE_train_data,mean(AE_train_data,2));
% AE_train_data = bsxfun(@rdivide,AE_train_data,(max(AE_train_data,[],2)-min(AE_train_data,[],2)));

trainData = bsxfun(@minus,trainData,avg);
trainData = bsxfun(@rdivide,trainData,sd);

testData = bsxfun(@minus,testData,avg);
testData = bsxfun(@rdivide,testData,sd);

AE_train_data = bsxfun(@minus,AE_train_data,avg);
AE_train_data = bsxfun(@rdivide,AE_train_data,sd);



%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

AUTOENCODER =1;
 
addpath Autoencoder/

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta




addpath Autoencoder/

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

if AUTOENCODER
    
[sae1OptTheta, cost1] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, AE_train_data), ...
                              sae1Theta, options);


% -------------------------------------------------------------------------
fprintf('1st Autoencoder constructed \n');
W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
display_network(W1');

%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, AE_train_data);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta


[sae2OptTheta, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);


fprintf('2nd Autoencoder constructed \n');
% -------------------------------------------------------------------------
% W2 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
% display_network(W2')

%%======================================================================

%% Train the third sparse Autoencoder

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

sae3Theta = initializeParameters(hiddenSizeL3, hiddenSizeL2);

[sae3OptTheta, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL2, hiddenSizeL3, ...
                                   lambda, sparsityParam, ...
                                   beta, sae2Features), ...
                              sae3Theta, options);

 fprintf('3nd Autoencoder constructed \n');
                                    

%% Train the Fourth sparse Autoencoder

[sae3Features] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
                                        hiddenSizeL2, sae2Features);

sae4Theta = initializeParameters(hiddenSizeL4, hiddenSizeL3);

[sae4OptTheta, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL3, hiddenSizeL4, ...
                                   lambda, sparsityParam, ...
                                   beta, sae3Features), ...
                              sae4Theta, options);

 fprintf('4th Autoencoder constructed \n');
 
 
 %% Train the Fifth sparse Autoencoder

[sae4Features] = feedForwardAutoencoder(sae4OptTheta, hiddenSizeL4, ...
                                        hiddenSizeL4, sae3Features);

sae5Theta = initializeParameters(hiddenSizeL5, hiddenSizeL4);

[sae5OptTheta, cost2] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL4, hiddenSizeL3, ...
                                   lambda, sparsityParam, ...
                                   beta, sae4Features), ...
                              sae5Theta, options);

 fprintf('5th Autoencoder constructed \n');
 
 else
     
   load('saeOptTheta_masssey.mat');
     
 end
 
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
 
[sae3Features] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
                                        hiddenSizeL2, sae2Features);
                                    
[sae4Features] = feedForwardAutoencoder(sae4OptTheta, hiddenSizeL4, ...
                                        hiddenSizeL3, sae3Features);

[sae5Features] = feedForwardAutoencoder(sae5OptTheta, hiddenSizeL5, ...
                                        hiddenSizeL4, sae4Features);

%  Randomly initialize the parameters
% hiddenSizeL2 = hiddenSizeL2+1; %  i did it!!
% saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

addpath Softmax/

% lambda_softmax = 1e-4;
lambda_softmax = 1e-6;


trainFeatures = sae5Features;
softmaxModel = softmaxTrain(hiddenSizeL5, numClasses, lambda_softmax, ...
                            trainFeatures, trainLabels, options);

saeSoftmaxOptTheta = softmaxModel.optTheta(:);               
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
 stack = cell(5,1);

stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
                     hiddenSizeL3, hiddenSizeL2);
stack{3}.b = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);

stack{4}.w = reshape(sae4OptTheta(1:hiddenSizeL4*hiddenSizeL3), ...
                     hiddenSizeL4, hiddenSizeL3);
stack{4}.b = sae3OptTheta(2*hiddenSizeL4*hiddenSizeL3+1:2*hiddenSizeL4*hiddenSizeL3+hiddenSizeL4);

stack{5}.w = reshape(sae5OptTheta(1:hiddenSizeL5*hiddenSizeL4), ...
                     hiddenSizeL5, hiddenSizeL4);
stack{5}.b = sae3OptTheta(2*hiddenSizeL5*hiddenSizeL4+1:2*hiddenSizeL5*hiddenSizeL4+hiddenSizeL5);



% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL5, ...
                                              numClasses, netconfig, ...
                                              lambda_softmax, trainData, trainLabels),stackedAETheta,options);
                                          
                                                                                

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL4, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL4, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
% after fine tning I get accuracy of 72.55 %  kinect 2
% before finetuning I get 33%


% kinect 3 45.65 , 73.75 

% kinect 4 Before Finetuning Test Accuracy: 41.950%
%After Finetuning Test Accuracy: 75.350%





toc
