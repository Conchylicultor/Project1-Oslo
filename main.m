%% Main: Linear regression
% From Etienne Pot and Lucile Madoulaud

% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_regression.mat');

%% Some data visualization
% to spot the categorical data for instance

% boxplot(X_train); % Before normalization

%% Dividing into two groups
% one to train the models, the other one to test and evaluate the model
% selection and global results

% Separate 90%-10%
evaluationIdx = (crossvalind('Kfold', length(y_train), 10) == 1); % Select the final testing set

X_Evaluation = X_train(evaluationIdx,:);
Y_Evaluation = y_train(evaluationIdx,:); % Testing data to evaluate the accuracy of the model selection

X_GlobalSet  = X_train(~evaluationIdx,:);
Y_GlobalSet  = y_train(~evaluationIdx,:); % Training data for the three models

%% Dividing our training data for our three models

% Thanks to hist(y_train), we can see the limits of the three clusters
% that we will take.

% hist(y_train, 200);
% title('Output histogram');
% xlabel('Y value');
% ylabel('Nb of occurence');

% The limits are:
% * 0-4100 for cluster 1
% * 4101-9000 for cluster 2
% * 9000-+++ for cluster 3

model1Idx = Y_GlobalSet <= 4100;
model2Idx = bitand(Y_GlobalSet > 4100,Y_GlobalSet <= 9000);
model3Idx = Y_GlobalSet > 9000;

% Assure correctness
assert(length(Y_GlobalSet) == sum(model1Idx+model2Idx+model3Idx), 'Values in no model');

sum(model1Idx)
sum(model2Idx)
sum(model3Idx)

% Creation of our three training set
X_Model1 = X_GlobalSet(model1Idx,:);
y_Model1 = Y_GlobalSet(model1Idx);

X_Model2 = X_GlobalSet(model2Idx,:);
y_Model2 = Y_GlobalSet(model2Idx);

X_Model3 = X_GlobalSet(model3Idx,:);
y_Model3 = Y_GlobalSet(model3Idx);

% Lets visualise our data clouds

% First possibility: Y=f(X)
% figure(1);
% plot(X_Train, y_Train, '.r'); % Our data points "brut"

% All data points
% hold on;
% for i= 1:length(X_train(1,:))
%     figure(i);
%     hold on;
%     
%     plot(X_Model1(:,i), y_Model1, '.r');
%     plot(X_Model2(:,i), y_Model2, '.g');
%     plot(X_Model3(:,i), y_Model3, '.b');
% end

% TODO: Plot ambiguity zones in another color to see eventual correlations
% to select disriminate criteria

%% Train the models
% We use k-cross validation to extract the bests parametters for the three
% models

% TODO: does it works better if we do the same nomalisation for everyone ??

k=12; % Parametter for the cross validation
[beta1, mean1, std1] = trainRegressionModel(X_Model1, y_Model1, k, 1);
[beta2, mean2, std2] = trainRegressionModel(X_Model2, y_Model2, k, 2);
[beta3, mean3, std3] = trainRegressionModel(X_Model3, y_Model3, k, 3);

%% Select model for the testing data
% When ambiguitty, we compute weight the results of the two closest model

% TODO: When ambiguitty, we compute weight the results of the two closest model

% We decide the cluster from some variables
% figure(1);
% plot(X_train(:,16), y_train, '.');
% figure(2);
% plot(X_train(:,38), y_train, '.');

% We determine in which cluster we are
selectionCondition1 = X_Evaluation(:,16) > 15.0;
selectionCondition2 = X_Evaluation(:,38) < 15.5;

selectionModel1 = selectionCondition2;
selectionModel2 = bitand(~selectionCondition1, ~selectionCondition2);
selectionModel3 = selectionCondition1;

X_Model1 = X_Evaluation(selectionModel1,:);
y_Model1 = Y_Evaluation(selectionModel1);

X_Model2 = X_Evaluation(selectionModel2,:);
y_Model2 = Y_Evaluation(selectionModel2);

X_Model3 = X_Evaluation(selectionModel3,:);
y_Model3 = Y_Evaluation(selectionModel3);

% Model visualisation
figure(1);
hold on;
plot(X_Model1(:,16), y_Model1, '.r');
plot(X_Model2(:,16), y_Model2, '.g');
plot(X_Model3(:,16), y_Model3, '.b');
figure(2);
hold on;
plot(X_Model1(:,38), y_Model1, '.r');
plot(X_Model2(:,38), y_Model2, '.g');
plot(X_Model3(:,38), y_Model3, '.b');

% Normalize data according to the corresponding model (TODO: TO REMOVE IF WE NORMALIZE THE DATA AT ONCE)

for i = 1:length(X_train(1,:))
    X_Model1(:,i) = (X_Model1(:,i)-mean1(i))/std1(i);
    X_Model2(:,i) = (X_Model2(:,i)-mean2(i))/std2(i);
    X_Model3(:,i) = (X_Model3(:,i)-mean3(i))/std3(i);
end

% form tX
tX_Model1 = [ones(length(y_Model1), 1) X_Model1];
tX_Model2 = [ones(length(y_Model2), 1) X_Model2];
tX_Model3 = [ones(length(y_Model3), 1) X_Model3];

costRMSE(y_Model1, tX_Model1, beta1)
costRMSE(y_Model2, tX_Model2, beta2)
costRMSE(y_Model3, tX_Model3, beta3)

% TODO: Evaluate how well our model selection perform

return;

%% GARBAGE CODE: TO DELETE

% X_trainModel1 = X_train(X_train(:,16) > 15.0, :);
% y_trainModel1 = y_train(X_train(:,16) > 15.0);

% Collumn 63(cat 4), 28(cat 4)!!! and 10(cat 3) could help discriminate

% Warning: apply the same normalization that for the training data

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
