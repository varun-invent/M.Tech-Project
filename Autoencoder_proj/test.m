
% load AEOptTheta
% load net config

function out_class = test(test_vector)
inputSize = length(test_vector);
[out_class] = stackedAEPredict(stackedAEOptTheta, inputSize, 200, ...
         5, netconfig, test_vector);
end

% check for continuity of labels for atleast 3 times 
% Then if continuous send to serial..
% and perform action upto 4 seconds