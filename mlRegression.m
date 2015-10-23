function [ beta, costResult ] = mlRegression(testingSet, trainingSet)
%leastSquares Least squares using normal equations.

% Parametters
k_fold = 5; % Cross validation

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
tX_test = [ones(length(X_test(:,1)), 1) X_test]; % TODO: Or poly(X_test, degree) ???

costResult = zeros(k_fold,1);
for k = 1:k_fold % Cross validation
    % Generate train and test data
    kPermIdx = (Indices~=k); % Dividing in two groups
    
    tX_TrainSet = tX_train(kPermIdx,:);
    Y_TrainSet = y_train(kPermIdx,:);
    tX_TestSet  = tX_train(~kPermIdx,:);
    Y_TestSet  = y_train(~kPermIdx,:);
    
    
    
    % Machine learning: compute parametters
    beta = leastSquares(Y_TrainSet, tX_TrainSet);
    
    % Machine learning: make predictions
    costResult(k) = costRMSE(Y_TestSet, tX_TestSet, beta);
    
    % Save predictions
end

%boxplot(costResult);

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

end
