function [costTraining, costTesting] = mlRegression(X_train, y_train, X_test, k_param)
%leastSquares Least squares using normal equations.

% Parametters
k_fold = k_param; % Cross validation

%plot(X_train, y_train);
%boxplot(X_train); % Before normalization

% Randomly permute the data
idx = randperm(length(y_train)); % Training
X_train = X_train(idx,:);
y_train = y_train(idx);

idx = randperm(length(X_test(:,1))); % Testing
X_test = X_test(idx,:);

% Normalizing the data
% TODO: Do we have to normalized binary data (0.85, -1.17) ???
% TODO: Remove categorical data

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

% boxplot(X_train); % After normalization

% Cross validation (make the random permutaion useless)
Indices = crossvalind('Kfold', length(y_train), k_fold);

% form tX
tX_train = [ones(length(y_train), 1) X_train];
tX_test = [ones(length(X_test(:,1)), 1) X_test];

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
    betaLeastSquare = leastSquares(Y_TrainSet, tX_TrainSet);
    costTraining(k,1) = costRMSE(Y_TrainSet, tX_TrainSet, betaLeastSquare);
    costTesting(k,1) = costRMSE(Y_TestSet, tX_TestSet, betaLeastSquare);
    
    betaGradient = leastSquaresGD(Y_TrainSet, tX_TrainSet, 0.01);
    costTraining(k,2) = costRMSE(Y_TrainSet, tX_TrainSet, betaGradient);
    costTesting(k,2) = costRMSE(Y_TestSet, tX_TestSet, betaGradient);
    
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

figure(1);
%hist(tX_TestSet*betaLeastSquare, 50);
hist(tX_TestSet*betaGradient, 50);
%hist(tX_TestSet*betaRidge, 50);
figure(2);
boxplot([costTraining costTesting]);

% for randomly sort the data x times
%   for divide the data in k part
%       for dimention of the regretion (first degree, second degree ?)
%           for cross validation (compute value of lambda)
% TODO: Estimate complexity

end
