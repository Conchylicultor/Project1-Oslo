function [beta, cost] = trainRegressionModel(X_train, y_train, k_fold, idModel)
%trainRegressionModel Find the best parametters beta using cross validation

disp(['Compute regression for model ',num2str(idModel)]);

% Randomly permute the data
idx = randperm(length(y_train)); % Training
X_train = X_train(idx,:);
y_train = y_train(idx);

% Normalizing the data
% TODO: Do we have to normalized binary data (0.85, -1.17) ???
% TODO: Dummy encoding for categorical data

% Cross validation (DON'T make the random permutaion useless)
Indices = crossvalind('Kfold', length(y_train), k_fold);

% form tX
tX_train = [ones(length(y_train), 1) X_train];

costTraining = zeros(k_fold,3);
costTesting = zeros(k_fold,3);

valsLambda = logspace(-1,3,30);
costRidgeTraining = zeros(k_fold, length(valsLambda));
costRidgeTesting = zeros(k_fold, length(valsLambda));
    
for k = 1:k_fold % Cross validation
    % Generate train and test data
    kPermIdx = (Indices~=k); % Dividing in two groups
    
    tX_TrainSet = tX_train(kPermIdx,:);
    Y_TrainSet = y_train(kPermIdx,:);
    tX_TestSet  = tX_train(~kPermIdx,:);
    Y_TestSet  = y_train(~kPermIdx,:);
    
    % Machine learning: compute parametters and make predictions
%     betaLeastSquare = leastSquares(Y_TrainSet, tX_TrainSet);
%     costTraining(k,1) = costRMSE(Y_TrainSet, tX_TrainSet, betaLeastSquare);
%     costTesting(k,1) = costRMSE(Y_TestSet, tX_TestSet, betaLeastSquare);
%     
%     betaGradient = leastSquaresGD(Y_TrainSet, tX_TrainSet, 0.01);
%     costTraining(k,2) = costRMSE(Y_TrainSet, tX_TrainSet, betaGradient);
%     costTesting(k,2) = costRMSE(Y_TestSet, tX_TestSet, betaGradient);
    
    for i = 1:length(valsLambda)
        lambda = valsLambda(i);
        betaRidge = ridgeRegression(Y_TrainSet, tX_TrainSet, lambda);
    	costRidgeTraining(k,i) = costRMSE(Y_TrainSet, tX_TrainSet, betaRidge);
    	costRidgeTesting(k,i) = costRMSE(Y_TestSet, tX_TestSet, betaRidge);
    end
    
    % Save predictions
end

%% Compute the cost for the ridge regression (and extract the right value of beta)

meanCostRidge = mean(costRidgeTesting);
[~, minCostRidgeIdx] = min(meanCostRidge);

%valsLambda(minCostRidgeIdx) % Debug: Best value of lambda

costTraining(:,3) = costRidgeTraining(:,minCostRidgeIdx);
costTesting(:,3) = costRidgeTesting(:,minCostRidgeIdx);

betaRidge = ridgeRegression(Y_TrainSet, tX_TrainSet, valsLambda(minCostRidgeIdx)); % Recompute beta again with the best value of lambda


%% Plot some results (compare different methods)

% figure(idModel*1000 + 1);
% %hist(tX_TestSet*betaLeastSquare, 50);
% %hist(tX_TestSet*betaGradient, 50);
% hist(tX_TestSet*betaRidge);
% figure(idModel*1000 + 2);
% boxplot([costTraining costTesting]);

beta = betaRidge;
cost = mean(costTesting(:, 3));

disp(['Cost model ', num2str(idModel), ': ', num2str(cost)]);

% figure(3);
% plot(tX_TestSet, tX_TestSet*betaRidge, '.g');
% figure(4);
% plot(tX_TestSet, tX_TestSet*betaRidge - Y_TestSet, '.g');
% figure(5);
% plot(tX_TestSet, Y_TestSet, '.r');

% for randomly sort the data x times
%   for divide the data in k part
%       for dimention of the regretion (first degree, second degree ?)
%           for cross validation (compute value of lambda)
% TODO: Estimate complexity

end
