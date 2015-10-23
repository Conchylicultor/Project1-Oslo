% Clear workspace
clc;
clear all;
close all;

% Loading data
disp('Project1 - Oslo Team');
load('Oslo_regression.mat');

%% Visualize Data
% Visualize Y=f(X), allow us to see some correlation
NbColor = 5;
colorMap = hsv(NbColor);
for i= 1:length(X_train(1,:))
    figure(floor((i-1)/NbColor) + 1);
    hold on;
    plot(X_train(:,i), y_train, '.', 'Color',colorMap(mod(i-1, NbColor) + 1,:));
end

% We see here three clusters of points
hist(y_train, 200);

%%

% Extract collums which allow us to discriminate between model
% Determine which model to apply for each X value

if model1
    % Extract good columns for the model
    % Make eventual transformation
    % Compute beta value with cross validation
    
    % Use the model on our training value
elseif model2
    
elseif model3
end
