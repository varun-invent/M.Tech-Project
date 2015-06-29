function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
% cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% data = [ones(1,size(data,2)); data];



M = theta*data; % k x m
M = bsxfun(@minus, M, max(M, [], 1));
var1 = exp(M); % k x m
normalizers =  sum(var1); % 1 x m normalizers
hypothesis = log(bsxfun(@rdivide,var1,normalizers));% k x m  ~ h(x)
hypothesis_ = bsxfun(@rdivide,var1,normalizers);
var2 = groundTruth .* hypothesis ;

cost = -sum(sum(var2))/numCases ;

 thetagrad = -(1./numCases)*(groundTruth -hypothesis_)*data';


reg=  (lambda/2)*sum(sum(theta.^2));


cost = cost + reg;


% var3 = groundTruth -hypothesis_;
% var4=0;
% for j=1:numClasses
%     for m =1:numCases
%        var4= var4 + data(:,j) * var3(j,m);
%     end
%     thetagrad(j,:) = -var4' ./ numCases ; 
% end




 thetagrad = thetagrad + lambda*theta;





% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

