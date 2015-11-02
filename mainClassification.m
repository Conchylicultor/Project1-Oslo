%% Main: Linear regression
% From Etienne Pot and Lucile Madoulaud

% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_classification.mat');

global final;
final = 0;

%% Some data visualization and model extraction

% Remove outliers
idToKeep = bitand(X_train(:,10) > -40, X_train(:,10) < 25);
idToKeep = bitand(idToKeep, X_train(:,2) > -40);
idToKeep = bitand(idToKeep, X_train(:,28) > -2900);
idToKeep = bitand(idToKeep, X_train(:,27) < 70);
X_train = X_train(idToKeep,:);
y_train = y_train(idToKeep);

% figure;
% boxplot(X_train); % Before normalization
% title('X before normalization');

tabulate(y_train); % Proportions (class 1: 75%, class 2: 25%)

% We visualize our input as the sum of two gaussian distribution
% for i=1:length(X_train(1,:))
%     figure(i*100);
%     subplot(2,1,1);
%     %plot(X_train(:,i), y_train, '.');
%     hist(X_test(:,i), 100);
%     
%     subplot(2,1,2);
%     hist(X_train(:,i), 100);
% end

modelSelectionIdx_TestAndTrain = X_train(:,12) >= -0.6;
modelSelectionIdx_TestAndJustTest = X_test(:,12) >= -0.6;

% for i=1:length(X_train(1,:))
%     figure(i*100);
%     subplot(2,2,1);
%     plot(X_Model1(:,i), y_Model1, '.');
%     
%     subplot(2,2,2);
%     hist(X_Model1(:,i), 100);
%     
%     subplot(2,2,3);
%     plot(X_Model2(:,i), y_Model2, '.');
%     
%     subplot(2,2,4);
%     hist(X_Model2(:,i), 100);
% end

% Correlation for each of the models

% figure;
% gplotmatrix(X_Model1, X_Model1, y_Model1);
% 
% figure;
% gplotmatrix(X_Model2, X_Model2, y_Model2);


%% Data transformation
% Normalisation, dummy encoding

[ X_train, X_test ] = dataTransform(X_train, X_test); % Dummy encoding & cie

% % Only keeping necessary column
% X_train = [X_train(:,2) X_train(:,6) X_train(:,16)];
% X_test  = [X_test(:,2)  X_test(:,6)  X_test(:,16)];

meanX = zeros(1, length(X_train(1,:)));
stdX = zeros(1, length(X_train(1,:)));
for i = 1:length(X_train(1,:))
    %if sum(mod(X_train(:,i),1)) ~= 0 % Non categorical data
        meanX(i) = mean(X_train(:,i));
        stdX(i) = std(X_train(:,i));

        X_train(:,i) = (X_train(:,i)-meanX(i))/stdX(i);

        % We normalize our testing data with the same value that for our testing
        % data (using the same mean and std that for the training)
        X_test(:,i) = (X_test(:,i)-meanX(i))/stdX(i);
    %end
end

%figure(10);
%boxplot(X_train); % After normalization

% TODO: Remove those collumns from the model testing ?

%% Dividing into two groups
% one to train the models, the other one to test and evaluate the model
% selection and global results

if final
    X_Evaluation = X_test; % Testing data
    
    model1_Idx_Evaluation = modelSelectionIdx_TestAndJustTest;
    model2_Idx_Evaluation = ~modelSelectionIdx_TestAndJustTest;
    
    model1_Idx_GlobalSet = modelSelectionIdx_TestAndTrain;
    model2_Idx_GlobalSet = ~modelSelectionIdx_TestAndTrain; % Taking all training
    
    X_GlobalSet  = X_train;
    Y_GlobalSet  = y_train; % Training data for the two models
    
%     evaluationIdx = (crossvalind('Kfold', length(y_train), 10) == 1); % Select the final testing set
% 
%     ModelIdx_GlobalSet = modelSelectionIdx_TestAndTrain(~evaluationIdx); % Which model apply
%     model1_Idx_GlobalSet = ModelIdx_GlobalSet;
%     model2_Idx_GlobalSet = ~ModelIdx_GlobalSet;
% 
%     X_GlobalSet  = X_train(~evaluationIdx,:);
%     Y_GlobalSet  = y_train(~evaluationIdx,:);
else
    % Separate 90%-10%
    evaluationIdx = (crossvalind('Kfold', length(y_train), 10) == 1); % Select the final testing set

    ModelIdx_Evaluation = modelSelectionIdx_TestAndTrain(evaluationIdx);
    ModelIdx_GlobalSet = modelSelectionIdx_TestAndTrain(~evaluationIdx); % Which model apply
    
    model1_Idx_Evaluation = ModelIdx_Evaluation;
    model2_Idx_Evaluation = ~ModelIdx_Evaluation;
    
    model1_Idx_GlobalSet = ModelIdx_GlobalSet;
    model2_Idx_GlobalSet = ~ModelIdx_GlobalSet;
    
    X_Evaluation = X_train(evaluationIdx,:);
    Y_Evaluation = y_train(evaluationIdx,:); % Testing data to evaluate the accuracy of the model selection

    X_GlobalSet  = X_train(~evaluationIdx,:);
    Y_GlobalSet  = y_train(~evaluationIdx,:); % Training data for the three models

    % For choosing the right model
    % S_Evaluation = xModelSelectionTrain(evaluationIdx, :);
    % S_GlobalSet  = xModelSelectionTrain(~evaluationIdx, :);

    % To be sure
    % assert(length(S_GlobalSet(:,1)) == length(Y_GlobalSet), 'Error: more or less values for the model selection');
end

%% Dividing our training data for our three models

lengthModel1 = sum(model1_Idx_GlobalSet);
lengthModel2 = sum(model2_Idx_GlobalSet);

% Size/proportion of each model
disp(['Model1 : ', num2str(lengthModel1), ' (', num2str(lengthModel1/length(Y_GlobalSet)*100),'%)']);
disp(['Model2 : ', num2str(lengthModel2), ' (', num2str(lengthModel2/length(Y_GlobalSet)*100),'%)']);

% Assure correctness
assert(length(Y_GlobalSet) == sum(model1_Idx_GlobalSet+model2_Idx_GlobalSet), 'Values in no model');

% Creation of our three training set
X_Model1_Train = X_train(model1_Idx_GlobalSet, :);
y_Model1_Train = y_train(model1_Idx_GlobalSet);

X_Model2_Train = X_train(model2_Idx_GlobalSet, :);
y_Model2_Train = y_train(model2_Idx_GlobalSet);

% % Plot the correlations
% figure;
% gplotmatrix(X_Model1_Train, X_Model1_Train, y_Model1_Train);

% figure;
% gplotmatrix(X_Model2_Train, X_Model2_Train, y_Model2_Train);

%% Train the models
% We use k-cross validation to extract the bests parametters for the three
% models

disp('-------------------------------------------------------');

global allTrainingCost;
global allTestingCost;

%allK = 2:20;% Parametter for the cross validation
allK = 5:5; % For final, the k value does not count

allTrainingCost = zeros(3, length(allK));
allTestingCost = zeros(3, length(allK));
for k=allK
    [beta1] = trainClassificationModel(X_Model1_Train, y_Model1_Train, k, 1);
    [beta2] = trainClassificationModel(X_Model2_Train, y_Model2_Train, k, 2);
end

costClass( y_Model1_Train, [ones(length(y_Model1_Train),1) X_Model1_Train], beta1)
costClass( y_Model2_Train, [ones(length(y_Model2_Train),1) X_Model2_Train], beta2)

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

% Model selection

X_Model1_Test = X_Evaluation(model1_Idx_Evaluation,:);
X_Model2_Test = X_Evaluation(model2_Idx_Evaluation,:);

lengthModel1 = length(X_Model1_Test(:,1));
lengthModel2 = length(X_Model2_Test(:,1));
lengthTotal = sum(model1_Idx_Evaluation) + sum(model2_Idx_Evaluation);

disp(['Model1 (guess) : ', num2str(lengthModel1), ' (', num2str(lengthModel1/lengthTotal*100),'%)']);
disp(['Model2 (guess) : ', num2str(lengthModel2), ' (', num2str(lengthModel2/lengthTotal*100),'%)']);

if ~final
    y_Model1_Test = Y_Evaluation(model1_Idx_Evaluation);
    y_Model2_Test = Y_Evaluation(model2_Idx_Evaluation);
end

% Model visualisation

% Assure that the models are correctly separated
figure(60);
subplot(2,1,1);
hist(X_Model1_Test(:,10), 50);
subplot(2,1,2);
hist(X_Model2_Test(:,10), 50);

% form tX
tX_Model1_Test = [ones(length(X_Model1_Test(:,1)), 1) X_Model1_Test];
tX_Model2_Test = [ones(length(X_Model2_Test(:,1)), 1) X_Model2_Test];

if ~final
    % Compute individual error for each model
    disp(['Model 1 perfs: ' , num2str(costClass(y_Model1_Test, tX_Model1_Test, beta1))]);
    disp(['Model 2 perfs: ' , num2str(costClass(y_Model2_Test, tX_Model2_Test, beta2))]);

    % Data visualisation
   
    figure(61);
    map = [1, 0, 0
           1, 0.6, 0
           1, 0.6, 0
           1, 0.6, 0
           0, 1, 0
           0.8, 0.6, 0.5
           0.8, 0.6, 0.5
           0.8, 0.6, 0.5
           0, 0, 1];
    colormap(map);
    subplot(2,2,1);
    scatter(X_Model1_Test(:,16), X_Model1_Test(:,2), 2, y_Model1_Test);
    title('Ground truth');
    subplot(2,2,2);
    scatter(X_Model1_Test(:,16), X_Model1_Test(:,2), 2, sigmoid(tX_Model1_Test*beta1) > 0.5);
    title('Prediction');
    subplot(2,2,3);
    scatter(X_Model1_Test(:,10), X_Model1_Test(:,8), 2, y_Model1_Test);
    title('Ground truth 2');
    subplot(2,2,4);
    scatter(X_Model1_Test(:,10), X_Model1_Test(:,8), 2, sigmoid(tX_Model1_Test*beta1) > 0.5);
    title('Prediction 2');

    % Compute the global cost

    % WARNING: When we will do the averaged prediction, we need to make
    % sure that we don't count two times the outliers in our count cost

    finalCost = costClass(y_Model1_Test, tX_Model1_Test, beta1) * length(y_Model1_Test) + ...
                costClass(y_Model2_Test, tX_Model2_Test, beta2) * length(y_Model2_Test);

    finalCost = finalCost / (length(y_Model1_Test) + length(y_Model2_Test));

    disp(['Final error: ' , num2str(finalCost)]);

    % We record the cost to make statistics
    global currentCost;
    currentCost = finalCost;

end

y_Test_Model1 = +(sigmoid(tX_Model1_Test*beta1) > 0.5);
y_Test_Model1(y_Test_Model1 == 0) = -1;

y_Test_Model2 = +(sigmoid(tX_Model2_Test*beta2) > 0.5);
y_Test_Model2(y_Test_Model2 == 0) = -1;

% histogram of the predicted values
tabulate ( [y_Test_Model1 ; y_Test_Model2]);

y_Test_Model1 = sigmoid(tX_Model1_Test*beta1);
y_Test_Model2 = sigmoid(tX_Model2_Test*beta2);

%% Make the final predictions and recording
% Some verifications about the consistency of the testing set

if final
    % Restore the results in the right order
    compteurModel1 = 1;
    compteurModel2 = 1;
    y_Final=zeros(length(modelSelectionIdx_TestAndJustTest),1);
    for i=1:length(modelSelectionIdx_TestAndJustTest)
        if modelSelectionIdx_TestAndJustTest(i) == 1
            y_Final(i) = y_Test_Model1(compteurModel1);
            compteurModel1 = compteurModel1 +1;
        elseif modelSelectionIdx_TestAndJustTest(i) == 0
            y_Final(i) = y_Test_Model2(compteurModel2);
            compteurModel2 = compteurModel2 +1;
        else
            disp('ERROR: UNKOWN MODEL');
        end
    end

    assert(compteurModel1 == length(y_Test_Model1) + 1);
    assert(compteurModel2 == length(y_Test_Model2) + 1);

    csvwrite('predictions_classification.csv', y_Final);
end

% Ending program
disp('Thanks for using our script');

return;

