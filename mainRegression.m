%% Main: Linear regression
% From Etienne Pot and Lucile Madoulaud

% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_regression.mat');

global final;
final = 0;

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
% figure(5);
% scatter3(X_train(:,16), X_train(:,38), y_train);

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
xModelSelectionTrain = [X_train(:,16) X_train(:,38)];
xModelSelectionTest = [X_test(:,16) X_test(:,38)];
%xModelSelection = [X_train(:,16).^3 X_train(:,38).^3 ];
%xModelSelectionTest = [X_test(:,16).^3  X_test(:,38).^3 ];

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
%boxplot(X_train); % After normalization

% TODO: Remove those collumns from the model testing ?

%% Dividing into two groups
% one to train the models, the other one to test and evaluate the model
% selection and global results

if final
    X_Evaluation = X_test; % Testing data

    X_GlobalSet  = X_train;
    Y_GlobalSet  = y_train; % Training data for the three models

    % For choosing the right model
    S_Evaluation = xModelSelectionTest;
    S_GlobalSet  = xModelSelectionTrain;
else
    % Separate 90%-10%
    evaluationIdx = (crossvalind('Kfold', length(y_train), 10) == 1); % Select the final testing set

    X_Evaluation = X_train(evaluationIdx,:);
    Y_Evaluation = y_train(evaluationIdx,:); % Testing data to evaluate the accuracy of the model selection

    X_GlobalSet  = X_train(~evaluationIdx,:);
    Y_GlobalSet  = y_train(~evaluationIdx,:); % Training data for the three models

    % For choosing the right model
    S_Evaluation = xModelSelectionTrain(evaluationIdx, :);
    S_GlobalSet  = xModelSelectionTrain(~evaluationIdx, :);

    % To be sure
    assert(length(S_GlobalSet(:,1)) == length(Y_GlobalSet), 'Error: more or less values for the model selection');
end

%% Dividing our training data for our three models

% Thanks to hist(y_train), we can see the limits of the three clusters
% that we will take.

% hist(y_train, 200);
% title('Output histogram');
% xlabel('Y value');
% ylabel('Nb of occurence');

% The limits are:
% * 0-4100 for cluster 1
% * 4101-8800 for cluster 2
% * 8800-+++ for cluster 3

model1Idx = Y_GlobalSet <= 4100;
model2Idx = bitand(Y_GlobalSet > 4100,Y_GlobalSet <= 8800);
model3Idx = Y_GlobalSet > 8800;

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
X_Model1_Train = X_GlobalSet(model1Idx,:);
y_Model1_Train = Y_GlobalSet(model1Idx);

X_Model2_Train = X_GlobalSet(model2Idx,:);
y_Model2_Train = Y_GlobalSet(model2Idx);

X_Model3_Train = X_GlobalSet(model3Idx,:);
y_Model3_Train = Y_GlobalSet(model3Idx);


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

% figure(11);
% hold on;
% scatter3(S_GlobalSet(model1Idx,1), S_GlobalSet(model1Idx,2), y_Model1, '.r');
% scatter3(S_GlobalSet(model2Idx,1), S_GlobalSet(model2Idx,2), y_Model2, '.g');
% scatter3(S_GlobalSet(model3Idx,1), S_GlobalSet(model3Idx,2), y_Model3, '.b');

% TODO: Plot ambiguity zones in another color to see eventual correlations
% to select disriminate criteria

%% Train the models
% We use k-cross validation to extract the bests parametters for the three
% models

disp('-------------------------------------------------------');

global allTrainingCost;
global allTestingCost;

%allK = 2:20;% Parametter for the cross validation
allK = 15:15; % For final, the k value does not count

allTrainingCost = zeros(3, length(allK));
allTestingCost = zeros(3, length(allK));
for k=allK
    [beta1] = trainRegressionModel(X_Model1_Train, y_Model1_Train, k, 1);
    [beta2] = trainRegressionModel(X_Model2_Train, y_Model2_Train, k, 2);
    [beta3] = trainRegressionModel(X_Model3_Train, y_Model3_Train, k, 3);
end

% Plot the learning curve
% figure (100);
% for i = 1:3
%     subplot(2,2,i);
%     hold on;
%     datasetSize = 100 - 100./allK; % Size of the training set
%     plot(datasetSize, allTrainingCost(i,:));
%     plot(datasetSize, allTestingCost(i,:));
%     title(['Learning curve, Model ', num2str(i)]);
%     xlabel('Training set size (%)');
%     ylabel('RMSE');
%     legend('Training RMSE', 'Testing RMSE');
% end

% Test data reduction

% Initialization
% if ~exist('cost1')
%     cost1 = zeros(1, length(X_Model1(1,:)));
%     cost2 = zeros(1, length(X_Model1(1,:)));
%     cost3 = zeros(1, length(X_Model1(1,:)));
% end

% for i = 1:length(X_Model1(1,:))
%     disp(num2str(i));
%     
%     X_Model1Red = X_Model1;
%     X_Model2Red = X_Model2;
%     X_Model3Red = X_Model3;
%     
%     X_Model1Red(:,i) = [];
%     X_Model2Red(:,i) = [];
%     X_Model3Red(:,i) = [];
%     
%     [beta1, cost1New(i)] = trainRegressionModel(X_Model1, y_Model1, k, 1);
%     [beta2, cost2New(i)] = trainRegressionModel(X_Model2, y_Model2, k, 2);
%     [beta3, cost3New(i)] = trainRegressionModel(X_Model3, y_Model3, k, 3);
% end
% figure;
% hold on;
% cost1 = cost1 + cost1New;
% cost2 = cost2 + cost2New;
% cost3 = cost3 + cost3New;
% plot(cost1);
% plot(cost2);
% plot(cost3);


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

% Model selection
selectionModel1 = modelSelectionIdx==1;
selectionModel2 = modelSelectionIdx==2;
selectionModel3 = modelSelectionIdx==3;
selectionModelOther = ~bitor(selectionModel1, bitor(selectionModel2, selectionModel3)); % Ambiguity cases

lengthModel1 = sum(selectionModel1);
lengthModel2 = sum(selectionModel2);
lengthModel3 = sum(selectionModel3);

disp(['Model1 (guess) : ', num2str(lengthModel1), ' (', num2str(lengthModel1/length(modelSelectionIdx)*100),'%)']);
disp(['Model2 (guess) : ', num2str(lengthModel2), ' (', num2str(lengthModel2/length(modelSelectionIdx)*100),'%)']);
disp(['Model3 (guess) : ', num2str(lengthModel3), ' (', num2str(lengthModel3/length(modelSelectionIdx)*100),'%)']);


if ~final
    % Ground truth
    selectionModel1Truth = Y_Evaluation <= 4100;
    selectionModel2Truth = bitand(Y_Evaluation > 4100,Y_Evaluation <= 8800);
    selectionModel3Truth = Y_Evaluation > 8800;
    
    % Model selection evaluation
    modelSelectionPerf = sum(abs(selectionModel1Truth - selectionModel1)) + ...
                         sum(abs(selectionModel2Truth - selectionModel2)) + ...
                         sum(abs(selectionModel3Truth - selectionModel3));
    modelSelectionPerf = modelSelectionPerf * 100 / length(Y_Evaluation);

    disp(['Selection performance error: ', num2str(modelSelectionPerf), '%']);
end

X_Model1_Test = X_Evaluation(selectionModel1,:);
X_Model2_Test = X_Evaluation(selectionModel2,:);
X_Model3_Test = X_Evaluation(selectionModel3,:);

X_ModelO_Test = X_Evaluation(selectionModelOther,:); % For outliers, we could try with more features

if ~final
    y_Model1_Test = Y_Evaluation(selectionModel1);
    y_Model2_Test = Y_Evaluation(selectionModel2);
    y_Model3_Test = Y_Evaluation(selectionModel3);
    
    y_ModelO_Test = Y_Evaluation(selectionModelOther);
end

outliersDetection(X_Model1_Train, X_Model1_Test);
outliersDetection(X_Model2_Train, X_Model2_Test);
outliersDetection(X_Model3_Train, X_Model3_Test); % Unfortunatly not complete

% Model visualisation

% figure(60);
% hold on;
% plot(S_Evaluation(selectionModel1,1), S_Evaluation(selectionModel1,2), '.r');
% plot(S_Evaluation(selectionModel2,1), S_Evaluation(selectionModel2,2), '.g');
% plot(S_Evaluation(selectionModel3,1), S_Evaluation(selectionModel3,2), '.b');
% plot(S_Evaluation(selectionModelOther,1), S_Evaluation(selectionModelOther,2), '.m');

% 3D Visualization of our clusters (see wrong classification)
% figure(62);
% hold on;
% scatter3(S_Evaluation(selectionModel1,1), S_Evaluation(selectionModel1,2), y_Model1_Test, '.r');
% scatter3(S_Evaluation(selectionModel2,1), S_Evaluation(selectionModel2,2), y_Model2_Test, '.g');
% scatter3(S_Evaluation(selectionModel3,1), S_Evaluation(selectionModel3,2), y_Model3_Test, '.b');
% scatter3(S_Evaluation(selectionModelOther,1), S_Evaluation(selectionModelOther,2), y_ModelO_Test, '.m');

% if sum(selectionModelOther) ~= 0
%     figure(61);
%     plot(S_Evaluation(selectionModelOther,1), S_Evaluation(selectionModelOther,2), '.m');
% end

% form tX
tX_Model1_Test = [ones(length(X_Model1_Test(:,1)), 1) X_Model1_Test];
tX_Model2_Test = [ones(length(X_Model2_Test(:,1)), 1) X_Model2_Test];
tX_Model3_Test = [ones(length(X_Model3_Test(:,1)), 1) X_Model3_Test];

if ~final
    % Compute individual error for each model
    disp(['Model 1 perfs: ' , num2str(costRMSE(y_Model1_Test, tX_Model1_Test, beta1))]);
    disp(['Model 2 perfs: ' , num2str(costRMSE(y_Model2_Test, tX_Model2_Test, beta2))]);
    disp(['Model 3 perfs: ' , num2str(costRMSE(y_Model3_Test, tX_Model3_Test, beta3))]);

    % Data visualisation

    figure(60);
    subplot(1,2,1);
    hold on;
    plot(X_Model1_Test(:,16), y_Model1_Test, '.r');
    plot(X_Model2_Test(:,16), y_Model2_Test, '.g');
    plot(X_Model3_Test(:,16), y_Model3_Test, '.b');
    title('Ground truth');
    
    subplot(1,2,2);
    hold on;
    plot(X_Model1_Test(:,16), abs(y_Model1_Test - tX_Model1_Test*beta1), '.r');
    plot(X_Model2_Test(:,16), abs(y_Model2_Test - tX_Model2_Test*beta2), '.g');
    plot(X_Model3_Test(:,16), abs(y_Model3_Test - tX_Model3_Test*beta3), '.b');
    title('Prediction errors');
    xlabel('X');
    ylabel('Error (abs(y-tX*beta))');

    % Compute the global cost

    % WARNING: When we will do the averaged prediction, we need to make
    % sure that we don't count two times the outliers in our count cost

    finalCost = costMSE(y_Model1_Test, tX_Model1_Test, beta1) * length(y_Model1_Test) + ...
                costMSE(y_Model2_Test, tX_Model2_Test, beta2) * length(y_Model2_Test) + ...
                costMSE(y_Model3_Test, tX_Model3_Test, beta3) * length(y_Model3_Test);

    finalCost = finalCost / (length(y_Model1_Test) + length(y_Model2_Test) + length(y_Model3_Test));

    finalCost = sqrt(2*finalCost);
    disp(['Final rmse: ' , num2str(finalCost)]);

    % We record the cost to make statistics
    global currentCost;
    currentCost = finalCost;

end

y_Test_Model1 = tX_Model1_Test*beta1;
y_Test_Model2 = tX_Model2_Test*beta2;
y_Test_Model3 = tX_Model3_Test*beta3;

% histogram of the predicted values
figure;
hist ( [y_Test_Model1 ; y_Test_Model2 ; y_Test_Model3], 100);
title('Histogram of the predictions');
xlabel('Predicted Y');
ylabel('Frequency');

assert (sum(selectionModelOther) == 0, 'Warning: some variables are in no model');

%% Make the final predictions and recording
% Some verifications about the consistency of the testing set

if final
    % 3D Visualization of our prediction
    figure(500);
    hold on;
    scatter3(S_Evaluation(selectionModel1,1), S_Evaluation(selectionModel1,2), y_Test_Model1, '.r');
    scatter3(S_Evaluation(selectionModel2,1), S_Evaluation(selectionModel2,2), y_Test_Model2, '.g');
    scatter3(S_Evaluation(selectionModel3,1), S_Evaluation(selectionModel3,2), y_Test_Model3, '.b');

    % Restore the results in the right order
    compteurModel1 = 1;
    compteurModel2 = 1;
    compteurModel3 = 1;
    y_Final=zeros(length(modelSelectionIdx),1);
    for i=1:length(modelSelectionIdx)
        if modelSelectionIdx(i) == 1
            y_Final(i) = y_Test_Model1(compteurModel1);
            compteurModel1 = compteurModel1 +1;
        elseif modelSelectionIdx(i) == 2
            y_Final(i) = y_Test_Model2(compteurModel2);
            compteurModel2 = compteurModel2 +1;
        elseif modelSelectionIdx(i) == 3
            y_Final(i) = y_Test_Model3(compteurModel3);
            compteurModel3 = compteurModel3 +1;
        else
            disp('ERROR: UNKOWN MODEL');
        end
    end

    assert(compteurModel1 == length(y_Test_Model1) + 1);
    assert(compteurModel2 == length(y_Test_Model2) + 1);
    assert(compteurModel3 == length(y_Test_Model3) + 1);

    csvwrite('predictions_regression.csv', y_Final);
end

% Ending program
disp('Thanks for using our script');

return;
