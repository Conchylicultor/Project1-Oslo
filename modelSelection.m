function [ modelSelectionIdx ] = modelSelection(S_Evaluation, ...
                                                S_GlobalSet, ...
                                                modelIdx,...
                                                kNN_param)
%modelSelection Apply KNN to select the closest model for each 

% Plot imput data
figure (50);
%colormap(jet);
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
subplot(1,2,1);
scatter(S_GlobalSet(:,1) ,S_GlobalSet(:,2), 4, modelIdx);
title('Training model selection');
set(gca,'Color',[0.2 0.4 0.4]);

% KNN using the euclidian distance
predictionsIdx = knnsearch(S_GlobalSet, S_Evaluation, 'K', kNN_param);

modelSelectionIdx = median(modelIdx(predictionsIdx),2);
%modelSelectionIdx = sum(modelIdx(predictionsIdx), 2)./kNN_param; % We average the models (weighted algo)

assert (length(modelSelectionIdx(:,1)) == length(predictionsIdx(:,1)), 'Error while predicting the models idx');

% Plot prediction data
subplot(1,2,2);
scatter(S_Evaluation(:,1) ,S_Evaluation(:,2), 4, modelSelectionIdx);
title('Prediction model selection');
set(gca,'Color',[0.2 0.4 0.4]);

% Get outliers (eventually)
figure(51);
outliersIdx = mod(modelSelectionIdx,1) ~= 0;
plot(S_Evaluation(outliersIdx,1) ,S_Evaluation(outliersIdx,2), '.m');
title('Outliers points');
set(gca,'Color',[0.2 0.4 0.4]);

end

