% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_regression.mat');

%plot(X_train, y_train, '.');

% Normalizing the data
% TODO: Do we have to normalized binary data (0.85, -1.17) ???

meanX = zeros(1, length(X_train(1,:)));
stdX = zeros(1, length(X_train(1,:)));
for i = 1:length(X_train(1,:))
  meanX(i) = mean(X_train(:,i));
  stdX(i) = std(X_train(:,i));
  
  X_train(:,i) = (X_train(:,i)-meanX(i))/stdX(i);
  % We normalize our testing data with the same value that for our testing
  % data (using the same mean and std that for the training)
  X_test(:,i) = (X_test(:,i)-meanX(i))/stdX(i);
end

% Randomly sorting the data

% for randomly sort the data x times
%   for divide the data in k part
%       for dimention of the regretion (first degree, second degree ?)
%           for cross validation (compute value of lambda)
% TODO: Estimate complexity

% Divide the data in K-parts

% Compute a lot of RMSE

% Plot the histogram of all RMSE

% Revert the normalization to obtain the correct results

% Ending program
disp('Thanks for using our script');
