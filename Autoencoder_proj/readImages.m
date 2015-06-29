% Reads Gesture Images and their respective labels and Stores them into Data matrix 


function Data = readImages(directory)

%directory = 'C:\Users\Varun Kumar\Dropbox\Codes\Thesis\Data\Triesch_test\';

directory1 = strcat(directory,'*.jpeg');
srcFiles = dir(directory1);  % the folder in which ur images exists
no_samples = length(srcFiles);
% filename = strcat(directory,srcFiles(1).name);

% image=imread(filename);

% [row,col] = size(image);
row = 28;
col = 28;
feature_vector_length =row*col ; 


Data = zeros(feature_vector_length,no_samples);

for i = 1 : no_samples
    fprintf('Image %d is read\n',i);
    filename = strcat(directory,srcFiles(i).name);
    I = imread(filename);
    if size(I,1) ~= row
        I = imresize(I,[row,col]);
    end
    Data(:,i) = I(:) ; 
end
disp('Data matrix created');