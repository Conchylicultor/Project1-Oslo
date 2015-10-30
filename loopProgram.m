% Compute 200 costs to predict our model performance

clc;
clear all;
close all;

global currentCost;

forLoop= 1:200;
finalCostLoop = zeros(1, length(forLoop));

for insideLoop=forLoop
    main;
    finalCostLoop(insideLoop) = currentCost;
    assert(length(finalCostLoop) == length(forLoop));
end

disp('Final cost result:  mean:', [num2str(mean(finalCostLoop)), '  std:',num2str(std(finalCostLoop))]);

% Plot the result
hist(finalCostLoop, 40);
title('Prediction');
xlabel('Rmse');
ylabel('Nb of occurence');

% For the data reduction 
% cost1 = cost1-mean(cost1);
% cost2 = cost2-mean(cost2);
% cost3 = cost3-mean(cost3);
% 
% figure
% plot(cost1)
% plot(cost2)
% plot(cost3)
