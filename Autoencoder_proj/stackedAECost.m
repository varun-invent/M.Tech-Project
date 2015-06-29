function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient

softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


% first i will fing the cost of the softmax
% then I find the weigh decay patameters of the autoencoders

a1 = data;
      z2 = stack{1}.w * a1 + repmat(stack{1}.b,1,m);
      a2 = sigmoid(z2);  
       
      z3 = stack{2}.w * a2 + repmat(stack{2}.b,1,m);
      a3 = sigmoid(z3);
      
      z4 = stack{3}.w * a3 + repmat(stack{3}.b,1,m);
      a4 = sigmoid(z4); 
      
      z5 = stack{4}.w * a4 + repmat(stack{4}.b,1,m);
      a5 = sigmoid(z5);
      
      z6 = stack{5}.w * a5 + repmat(stack{5}.b,1,m);
      a6 = sigmoid(z6);

%       a3 = [ones(1,size(a3,2)); a3];

M = softmaxTheta * a6; % k x m
M = bsxfun(@minus, M, max(M, [], 1));
var1 = exp(M); % k x m
normalizers =  sum(var1); % 1 x m normalizers
hypothesis = log(bsxfun(@rdivide,var1,normalizers));% k x m  ~ h(x)
hypothesis_ = bsxfun(@rdivide,var1,normalizers);
var2 = groundTruth .* hypothesis ;

cost = -sum(sum(var2))/m ;

%  reg=  (lambda/2)*((sum(sum(stack{1}.w .^2)) + sum(sum(stack{2}.w .^2)))...
%       +sum(sum(softmaxTheta.^2)));
 
 reg= (lambda/2) * sum(sum(softmaxTheta .^2));
 cost = cost+reg; 

% now calculate the gradient for softmax layer

softmaxThetaGrad = -(1./m)*(groundTruth -hypothesis_)*a6';
softmaxThetaGrad = softmaxThetaGrad + lambda*softmaxTheta;

% softmaxThetaGrad = -(1./m)*(groundTruth -hypothesis_)*a4';
% softmaxThetaGrad = softmaxThetaGrad + lambda*softmaxTheta;

% now calculate the thetaGrad for the stacks using backpropogation


gradJ = softmaxTheta' * (groundTruth - hypothesis_);

delta6 = -gradJ .* sigmoidGrad(z6);

delta5 = (stack{5}.w'*delta6) .* sigmoidGrad(z5);

delta4 = (stack{4}.w'*delta5) .* sigmoidGrad(z4);

delta3 = (stack{3}.w'*delta4) .* sigmoidGrad(z3);

delta2 = (stack{2}.w'*delta3) .* sigmoidGrad(z2);

stackgrad{5}.w = (1/m) .* delta6*a5';

stackgrad{4}.w = (1/m) .* delta5*a4';

stackgrad{3}.w = (1/m) .* delta4*a3';

stackgrad{2}.w = (1/m) .* delta3*a2';% + lambda .* stack{2}.w;

stackgrad{1}.w = (1/m) .* delta2*a1';% + lambda .* stack{1}.w;

stackgrad{1}.b = (1/m) .* sum(delta2,2); 
  
stackgrad{2}.b = (1/m) .* sum(delta3,2); 

stackgrad{3}.b = (1/m) .* sum(delta4,2);

stackgrad{4}.b = (1/m) .* sum(delta5,2);

stackgrad{5}.b = (1/m) .* sum(delta6,2);
  

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigmGrad = sigmoidGrad(x)
     sigmGrad = sigmoid(x) .* (1-sigmoid(x));
end
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
