function [ modelSelectionIdx ] = modelSelection(S_Evaluation, ...
                                                S_GlobalSet, ...
                                                modelIdx,...
                                                kNN_param)
%modelSelection Apply KNN to select the closest model for each 

% Plot imput data
figure (50);
colormap(cool);
scatter(S_GlobalSet(:,1) ,S_GlobalSet(:,2), 4, modelIdx);

% KNN using the euclidian distance
predictionsIdx = knnsearch(S_GlobalSet, S_Evaluation, 'K', kNN_param);

modelSelectionIdx = median(modelIdx(predictionsIdx),2);
%modelSelectionIdx = sum(modelIdx(predictionsIdx), 2)./kNN_param; % We average the models (weighted algo)

assert (length(modelSelectionIdx(:,1)) == length(predictionsIdx(:,1)), 'Error while predicting the models idx');

% Plot prediction data
figure (51);
colormap(jet);
scatter(S_Evaluation(:,1) ,S_Evaluation(:,2), 4, modelSelectionIdx);
set(gca,'Color',[0.2 0.4 0.4]);

end

