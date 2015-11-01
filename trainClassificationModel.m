function [beta, cost] = trainClassificationModel(X_train, y_train, k_fold, idModel)
%trainClassificationModel Find the best parametters beta using cross validation

disp(['Compute classification for model: ', num2str(idModel)]);

% Randomly permute the data
idx = randperm(length(y_train)); % Training
X_train = X_train(idx,:);
y_train = y_train(idx);

% Data already normalized

% Form tX
tX_train = [ones(length(y_train), 1) X_train];

alpha = 0.005;

global final;
if(final)    
    beta = penLogisticRegression(y_train, tX_train, lambda);
    
    cost = 0;
    
    disp(['Train cost rmse: ',    num2str(costRMSE(y_train, tX_train, beta)),' computed with fixed lambda']);
    disp(['Train cost class: ',   num2str(costClass(y_train, tX_train, beta)),' computed with fixed lambda']);
    disp(['Train cost logClass: ',num2str(costLogClass(y_train, tX_train, beta)),' computed with fixed lambda']);
    
    return;
end
    
% Cross validation (DON'T make the random permutaion useless)
Indices = crossvalind('Kfold', length(y_train), k_fold);

costTraining = zeros(k_fold,6);
costTesting = zeros(k_fold,6);

if idModel == 1
    valsLambda = logspace(0,0.5,10);
elseif idModel == 2
    valsLambda = logspace(0,0.5,10);
end

costPenTraining = zeros(k_fold, length(valsLambda));
costPenTesting = zeros(k_fold, length(valsLambda));
    
minI = 1;
for i = 1:length(valsLambda)
    for k = (k_fold-2):k_fold % Cross validation
        disp(['K:', num2str(k)]);
        
        % Generate train and test data
        kPermIdx = (Indices~=k); % Dividing in two groups

        tX_TrainSet = tX_train(kPermIdx,:);
        Y_TrainSet = y_train(kPermIdx,:);
        tX_TestSet  = tX_train(~kPermIdx,:);
        Y_TestSet  = y_train(~kPermIdx,:);

        % Machine learning: compute parametters and make predictions

        % Logistic regression
        %betaLogisticRegression = logisticRegression(Y_TrainSet, tX_TrainSet, alpha);
        %betaIrls = penLogisticRegression(Y_TrainSet, tX_TrainSet, alpha, lambda);
        %betaIrls = IRLS(Y_TrainSet, tX_TrainSet);
    %     
    %     costTraining(k, 1) = costRMSE(Y_TrainSet, tX_TrainSet, betaIrls);
    %     costTraining(k, 2) = costClass(Y_TrainSet, tX_TrainSet, betaIrls);
    %     costTraining(k, 3) = costLogClass(Y_TrainSet, tX_TrainSet, betaIrls);
    %     
    %     costTesting(k, 1) = costRMSE(Y_TestSet, tX_TestSet, betaIrls);
    %     costTesting(k, 2) = costClass(Y_TestSet, tX_TestSet, betaIrls);
    %     costTesting(k, 3) = costLogClass(Y_TestSet, tX_TestSet, betaIrls);

        
        % Penalized logistic regression
        % TODO

        lambda = valsLambda(i);
        betaPenLambda = penLogisticRegression(Y_TrainSet, tX_TrainSet, alpha, lambda);
        costPenTraining(k,i) = costClass(Y_TrainSet, tX_TrainSet, betaPenLambda);
        costPenTesting(k,i) = costClass(Y_TestSet, tX_TestSet, betaPenLambda);

        % Save predictions
    end
    
    if( mean(costPenTesting(:,i)) < mean(costPenTesting(:,minI)) || minI==1 )
        minI = i;
        betaPen = betaPenLambda;
    end
    
    disp(['L: ', num2str(mean(costPenTesting(:,i)))]);
end

%% Compute the cost for the penalized logistic regression (and extract the right value of beta)

costPen = costPenTesting(:,minI);
disp(['Best value of lambda:', num2str(valsLambda(minI))]) % Best value of lambda

figure(idModel*1000 + 10);
semilogx(valsLambda, mean(costPenTesting),'-sb');
hold on
semilogx(valsLambda, mean(costPenTesting),'-sr');
grid on

costTesting(:,2) = costPen;


%% Plot some results (compare different methods)

resultPrediction = +(sigmoid(tX_TestSet*betaPen) > 0.5);
resultPrediction(resultPrediction == 0) = -1;
tabulate(resultPrediction);

%figure(idModel*1000 + 1);
%boxplot([costTraining costTesting]);

beta = betaPen;
%costRmseVar = mean(costTesting(:, 1));
costClassVar = mean(costTesting(:, 2));
%costLogClassVar = mean(costTesting(:, 3));

% % Learning curve
% global allTrainingCost;
% global allTestingCost;
% 
% allTrainingCost(idModel, k_fold-1) = mean(costTraining(:, 4));
% allTestingCost(idModel, k_fold-1) =  mean(costTesting(:, 4));

%disp(['Cost model ', num2str(idModel), ': ', num2str(costRmseVar) , ', std: ', num2str(std(costTesting(:, 1)))]);
disp(['Cost model ', num2str(idModel), ': ', num2str(costClassVar) , ', std: ', num2str(std(costTesting(:, 2)))]);
%disp(['Cost model ', num2str(idModel), ': ', num2str(costLogClassVar) , ', std: ', num2str(std(costTesting(:, 3)))]);

end
