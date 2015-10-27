%% Main: Linear regression

% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_regression.mat');

% % Without categorical data
% collumnIdToTake = [4 5 6 7 8 9 11 12 13 15 17 18 19 21 22 23 24 25 26 27 29 31 32 33 34 35 36 37 39 40 43 44 45 46 47 48 49 50 51 52 54 55 56 57 58 60 61 62 64 65];
% collumnIdToRemove = [1 2 3 10 14 16 20 28 30 38 41 42 53 59 63];
% collumnIdToTake = [16 38 collumnIdToTake];
% 
% X_trainClean = X_train(:,collumnIdToTake(1));
% for i = collumnIdToTake(2:end)
%     X_trainClean = [X_trainClean X_train(:,i)];
% end
% 
% X_train = X_trainClean;



% % Trying extract only on of th gaussian
% X_trainModel1 = X_train(X_train(:,16) > 15.0, :);
% y_trainModel1 = y_train(X_train(:,16) > 15.0);
% X_train = X_trainModel1(:,26);
% y_train = y_trainModel1;
% X_trainModel1 = X_train(y_trainModel1 > 8000);
% y_trainModel1 = y_train(y_trainModel1 > 8000);
% X_train = X_trainModel1;
% y_train = y_trainModel1;
% 
% figure(512);
% plot(X_train, y_train, '.');

%X_trainModel2 = X_train(X_train(:,16) < 15.0, :);
%plot(X_trainModel1(:,16), y_train(X_train(:,16) > 15.0), '.r');
%plot(X_trainModel2(:,16), y_train(X_train(:,16) < 15.0), '.b');



% Highly correlated input
%plot(X_train(:,38), y_train, '.r');
%plot(X_train(:,16), y_train, '.r');
%plot(X_train(:,61), y_train, '.b');
%plot(X_train(:,12), y_train, '.b');
%plot(X_train(:,26), y_train, '.b');
%plot(X_train(:,19), y_train, '.y');
%plot(X_train(:,11), y_train, '.y');
%plot(X_train(:,37), y_train, '.y');

X_train = [X_train(:,38) ...
           X_train(:,16) ...
           X_train(:,61) ...
           X_train(:,12) ...
           X_train(:,26) ...
           X_train(:,19) ...
           X_train(:,11) ...
           X_train(:,37)];


%% Visualize Data
% % Visualize Y=f(X), allow us to see some correlation
% NbColor = 5;
% colorMap = hsv(NbColor);
% for i= 1:length(X_train(1,:))
%     figure(floor((i-1)/NbColor) + 1);
%     hold on;
%     plot(X_train(:,i), y_train, '.', 'Color',colorMap(mod(i-1, NbColor) + 1,:));
% end
% 
% % We see here three clusters of points
% %hist(y_train, 200);

%% Bias vs Variance diagnostic: Testing and training error = f(size of dataset)

datasetSize = 2:30;
averCostTraining = zeros(length(datasetSize),1);
averCostTesting = zeros(length(datasetSize),1);
for k=datasetSize
    [costTraining, costTesting] = mlRegression(X_train, y_train, X_test, k);
    
    idx = 3; % We take the results of leastSquare, gradientDescent or ridgeRegression
    averCostTraining(k-1) = mean(costTraining(:,idx));
    averCostTesting(k-1) = mean(costTesting(:,idx));
end

figure (100);
hold on;
datasetSize = length(y_train) - length(y_train)./datasetSize; % Size of the training set
plot(datasetSize, averCostTraining);
plot(datasetSize, averCostTesting);
title('Learning curve');
xlabel('Training set size');
ylabel('RMSE');

%%

% Extract collums which allow us to discriminate between model
% Determine which model to apply for each X value

model=1;
if model==1
    % Extract good columns for the model
    % Make eventual transformation
    % Compute beta value with cross validation
    
    % Use the model on our training value
elseif model==2
    
elseif model==3
end

% Ending program
disp('Thanks for using our script');
