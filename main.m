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

% % Columns useful to select the model
% figure(1);
% plot(X_train(:,16), y_train, '.');
% figure(2);
% plot(X_train(:,38), y_train, '.');

% % Histogram of those collumn (train)
% figure(3);
% hist(X_train(:,16), 100);
% figure(4);
% hist(X_train(:,38), 100);
% figure(5);
% plot(X_train(:,16), X_train(:,38), '.');

% % Histogram of those collumn (test)
% figure(3);
% hist(X_test(:,16), 100);
% figure(4);
% hist(X_test(:,38), 100);
% figure(5);
% plot(X_test(:,16), X_test(:,38), '.');

%% Data transformation
% Normalisation, dummy encoding

% Random permutation here or later (for now in trainRegressionModel) ???

% We save our collumns for the model selection
xModelSelection = [X_train(:,16) X_train(:,38)];
xModelSelectionTest = [X_test(:,16) X_test(:,38)];

% Do we not normalize collumn 16 and 38 ???

[ X_train, X_test ] = dataTransform(X_train, X_test); % Dummy encoding & cie

meanX = zeros(1, length(X_train(1,:)));
stdX = zeros(1, length(X_train(1,:)));
for i = 1:length(X_train(1,:))
    if sum(mod(X_train(:,i),1)) ~= 0 % Non categorical data
        meanX(i) = mean(X_train(:,i));
        stdX(i) = std(X_train(:,i));

        X_train(:,i) = (X_train(:,i)-meanX(i))/stdX(i);

        % We normalize our testing data with the same value that for our testing
        % data (using the same mean and std that for the training)
        X_test(:,i) = (X_test(:,i)-meanX(i))/stdX(i);
    end
end

%figure(10);
boxplot(X_train); % After normalization

% TODO: Remove those collumns from the model testing ?

%% Dividing into two groups
% one to train the models, the other one to test and evaluate the model
% selection and global results

% Separate 90%-10%
evaluationIdx = (crossvalind('Kfold', length(y_train), 10) == 1); % Select the final testing set

X_Evaluation = X_train(evaluationIdx,:);
Y_Evaluation = y_train(evaluationIdx,:); % Testing data to evaluate the accuracy of the model selection

X_GlobalSet  = X_train(~evaluationIdx,:);
Y_GlobalSet  = y_train(~evaluationIdx,:); % Training data for the three models

% For choosing the right model
S_Evaluation = xModelSelection(evaluationIdx, :);
S_GlobalSet  = xModelSelection(~evaluationIdx, :);

% To be sure
assert(length(S_GlobalSet(:,1)) == length(Y_GlobalSet), 'Error: more or less values for the model selection');

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

% Size/proportion of each model

lengthModel1 = sum(model1Idx);
lengthModel2 = sum(model2Idx);
lengthModel3 = sum(model3Idx);

disp(['Model1 : ', num2str(lengthModel1), ' (', num2str(lengthModel1/length(Y_GlobalSet)*100),'%)']);
disp(['Model2 : ', num2str(lengthModel2), ' (', num2str(lengthModel2/length(Y_GlobalSet)*100),'%)']);
disp(['Model3 : ', num2str(lengthModel3), ' (', num2str(lengthModel3/length(Y_GlobalSet)*100),'%)']);

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

% % Plot X(16) vs X(38)
% figure(10);
% hold on;
% plot(X_Model1(:,16), X_Model1(:,38), '.r');
% plot(X_Model2(:,16), X_Model2(:,38), '.g');
% plot(X_Model3(:,16), X_Model3(:,38), '.b');

% TODO: Plot ambiguity zones in another color to see eventual correlations
% to select disriminate criteria

%% Train the models
% We use k-cross validation to extract the bests parametters for the three
% models

% TODO: does it works better if we do the same nomalisation for everyone ??

disp('-------------------------------------------------------');

k=12; % Parametter for the cross validation
[beta1] = trainRegressionModel(X_Model1, y_Model1, k, 1);
[beta2] = trainRegressionModel(X_Model2, y_Model2, k, 2);
[beta3] = trainRegressionModel(X_Model3, y_Model3, k, 3);

disp('-------------------------------------------------------');

%% Select model for the testing data
% We use kNN to determine in which cluster we are

% TODO: When ambiguitty, we compute weight the results of the two closest model

kNN_param = 3;

% We determine in which cluster we are

modelSelectionIdx = modelSelection(S_Evaluation, ...
                                   S_GlobalSet, ...
                                   model1Idx*1 + model2Idx*2 + model3Idx*3, ...
                                   kNN_param);
                               
selectionModel1 = modelSelectionIdx==1;
selectionModel2 = modelSelectionIdx==2;
selectionModel3 = modelSelectionIdx==3;
selectionModelOther = ~bitor(selectionModel1, bitor(selectionModel2, selectionModel3)); % Ambiguity cases

X_Model1 = X_Evaluation(selectionModel1,:);
y_Model1 = Y_Evaluation(selectionModel1);

X_Model2 = X_Evaluation(selectionModel2,:);
y_Model2 = Y_Evaluation(selectionModel2);

X_Model3 = X_Evaluation(selectionModel3,:);
y_Model3 = Y_Evaluation(selectionModel3);

% TODO: Handle those cases !!!
X_ModelO = X_Evaluation(selectionModelOther,:);
y_ModelO = Y_Evaluation(selectionModelOther);

% TODO: Evaluate how well our model selection perform

% Model visualisation

% figure(60);
% hold on;
% plot(S_Evaluation(selectionModel1,1), S_Evaluation(selectionModel1,2), '.r');
% plot(S_Evaluation(selectionModel2,1), S_Evaluation(selectionModel2,2), '.g');
% plot(S_Evaluation(selectionModel3,1), S_Evaluation(selectionModel3,2), '.b');
% plot(S_Evaluation(selectionModelOther,1), S_Evaluation(selectionModelOther,2), '.m');
% 
% if sum(selectionModelOther) ~= 0
%     figure(61);
%     plot(S_Evaluation(selectionModelOther,1), S_Evaluation(selectionModelOther,2), '.m');
% end

% form tX
tX_Model1 = [ones(length(y_Model1), 1) X_Model1];
tX_Model2 = [ones(length(y_Model2), 1) X_Model2];
tX_Model3 = [ones(length(y_Model3), 1) X_Model3];

% Compute individual error for each model
disp(['Model 1 perfs: ' , num2str(costRMSE(y_Model1, tX_Model1, beta1))]);
disp(['Model 2 perfs: ' , num2str(costRMSE(y_Model2, tX_Model2, beta2))]);
disp(['Model 3 perfs: ' , num2str(costRMSE(y_Model3, tX_Model3, beta3))]);

% Data visualisation

figure(60);
subplot(1,2,1);
hold on;
plot(X_Model1(:,16), y_Model1, '.r');
plot(X_Model2(:,16), y_Model2, '.g');
plot(X_Model3(:,16), y_Model3, '.b');
title('Ground truth');

subplot(1,2,2);
hold on;
plot(X_Model1(:,16), abs(y_Model1 - tX_Model1*beta1), '.r');
plot(X_Model2(:,16), abs(y_Model2 - tX_Model2*beta2), '.g');
plot(X_Model3(:,16), abs(y_Model3 - tX_Model3*beta3), '.b');
title('Prediction errors');

% Compute the global cost

% WARNING TODO: When we will do the averaged prediction, we need to make
% sure that we don't count two times the outliers in our count cost

finalCost = costMSE(y_Model1, tX_Model1, beta1) * length(y_Model1) + ...
            costMSE(y_Model2, tX_Model2, beta2) * length(y_Model2) + ...
            costMSE(y_Model3, tX_Model3, beta3) * length(y_Model3);
           
finalCost = finalCost / (length(y_Model1) + length(y_Model2) + length(y_Model3));

finalCost = sqrt(2*finalCost);
disp(['Final rmse: ' , num2str(finalCost)]);

% histogram of the predicted values
figure;
hist ( [tX_Model1*beta1 ; tX_Model2*beta2 ; tX_Model3*beta3], 100);

assert (sum(selectionModelOther) == 0, 'Warning: some variables are in no model');

%% Make the final predictions
% Some verifications about the consistency of the testing set

% modelSelectionIdxTest = modelSelection(xModelSelectionTest, ...
%                                        S_GlobalSet, ...
%                                        model1Idx*1 + model2Idx*2 + model3Idx*3, ...
%                                        kNN_param);
%                                    
% lengthModel1 = sum(modelSelectionIdxTest == 1);
% lengthModel2 = sum(modelSelectionIdxTest == 2);
% lengthModel3 = sum(modelSelectionIdxTest == 3);
% 
% disp(['Model1 (guess) : ', num2str(lengthModel1), ' (', num2str(lengthModel1/length(xModelSelectionTest(:,1))*100),'%)']);
% disp(['Model2 (guess) : ', num2str(lengthModel2), ' (', num2str(lengthModel2/length(xModelSelectionTest(:,1))*100),'%)']);
% disp(['Model3 (guess) : ', num2str(lengthModel3), ' (', num2str(lengthModel3/length(xModelSelectionTest(:,1))*100),'%)']);


% Ending program
disp('Thanks for using our script');

return;












%%
%% GARBAGE CODE: TO DELETE
%%
%%

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
