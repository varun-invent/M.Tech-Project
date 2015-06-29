function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% data = [ones(1,size(data,2)); data];

M = theta*data; % k x m
M = bsxfun(@minus, M, max(M, [], 1));
var1 = exp(M); % k x m
normalizers =  sum(var1); % 1 x m normalizers
hypothesis = log(bsxfun(@rdivide,var1,normalizers));% k x m  ~ h(x)

[~,pred] = max(hypothesis);



% ---------------------------------------------------------------------

end

