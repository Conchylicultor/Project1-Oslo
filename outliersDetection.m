function [ ] = outliersDetection( X_Train, X_Test )
%outliersDetection Is suppose to detect the outliers in the testing set

% Detailed explanation goes here

% Compute the mean of the data
meanPoint = mean(X_Train);

% Compute the distance to the mean of each data
distTrain = zeros(1,length(X_Train(:,1)));
for i=1:length(X_Train(:,1))
    distTrain(i) = norm(meanPoint - X_Train(i,:));
end

distTest = zeros(1,length(X_Test(:,1)));
for i=1:length(X_Test(:,1))
    distTest(i) = norm(meanPoint - X_Train(i,:));
end

% Plot histogram of the distance
% figure;
% subplot(2,1,1);
% hist(distTrain, 100);
% title('Dist from mean (train)');
% subplot(2,1,2);
% hist(distTest, 100);
% title('Dist from mean (test)');

end

