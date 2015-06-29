function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

m=size(data,2);

a1 = data;
 z2 = stack{1}.w * a1 + repmat(stack{1}.b,1,m);
      a2 = sigmoid(z2);  
       
      z3 = stack{2}.w * a2 + repmat(stack{2}.b,1,m);
      a3 = sigmoid(z3);
      
      z4 =  stack{3}.w * a3 + repmat(stack{3}.b,1,m);
      a4 = sigmoid(z4);
      
      z5 =  stack{4}.w * a4 + repmat(stack{4}.b,1,m);
      a5 = sigmoid(z5);
      
      z6 =  stack{5}.w * a5 + repmat(stack{5}.b,1,m);
      a6 = sigmoid(z6);
      
      
      a7 = softmaxTheta*a6; % k x m
a7 = bsxfun(@minus, a7, max(a7, [], 1));
var1 = exp(a7); % k x m
normalizers =  sum(var1); % 1 x m normalizers
hypothesis = bsxfun(@rdivide,var1,normalizers);% k x m  ~ h(x)

[~,pred] = max(hypothesis);










% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
