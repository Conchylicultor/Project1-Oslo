function [ X_TrainAfter, X_TestAfter ] = dataTransform(X_Train, X_Test)
%dataTransform Encode the categorical variables

% First: detect the categorical variables

% Initialisation
X_TrainAfter = zeros(length(X_Train(:,1)), 1);
X_TestAfter = zeros(length(X_Test(:,1)), 1);

for i=1:length(X_Train(1,:)) % For each collumn
    if sum(mod(X_Train(:,i),1) == 0)
        % disp([num2str(i), ' is categorical']);
        
        % Binary or categorical ?
        if (max(X_Train(:,i)) == 1)
            disp([num2str(i), ' is binary']);
            X_TrainAfter = [X_TrainAfter X_Train(:,i)];
            X_TestAfter = [X_TestAfter X_Test(:,i)]; % No modification for binary data
        else
            disp([num2str(i), ' is categorical (', num2str(max(X_Train(:,i))),')']);
            
            maxX = max(X_Train(:,i));
            minX = min(X_Train(:,i));
            
            assert (maxX >= max(X_Test(:,i)), 'Warning: Not the same dimention of the categorical data');
            assert (minX <= min(X_Test(:,i)), 'Warning: Not the same dimention of the categorical data');

            xCategoricalTrain = zeros(length(X_Train(:,i)), maxX - minX + 1 );
            xCategoricalTest  = zeros(length(X_Test(:,i)),  maxX - minX + 1 ); % Same dimention for both
            
            for j=minX:maxX
                % Check if there is a null column
                xCategoricalTrain(:,j-minX+1) = +(X_Train(:,i) == j);
                xCategoricalTest (:,j-minX+1) = +(X_Test(:,i) == j);
                
                assert(sum(xCategoricalTrain(:,j-minX+1)) ~= 0, 'Warning: null column');
                assert(sum(xCategoricalTest(:,j-minX+1)) ~= 0, 'Warning: null column');
            end
            
            % X_TrainAfter = [X_TrainAfter xCategoricalTrain];
            % X_TestAfter  = [X_TestAfter  xCategoricalTest];
        end
        
        
    else
        % We add the collumn and eventually apply a transformation
        X_TrainAfter = [X_TrainAfter basicMath(X_Train(:,i))];
        X_TestAfter = [X_TestAfter basicMath(X_Test(:,i))];
    end
end

% We remove the first collumn
X_TrainAfter = X_TrainAfter(:, 2:end);
X_TestAfter = X_TestAfter(:, 2:end);

end

function [Y] = basicMath(X)
    %Y = X;
    %Y = X.*X; % Much better
    Y = X.*X.*X; % Much much better
    %Y = X.*X.*X.*X;
    
    %Y = exp(X); % Worst
    %Y = 1./X; % If X = 0 ???
    
%     if(abs(X) == X) % Only positive values
%         %Y = sqrt(X); % If X < 0 ???
%         %Y = log(X); % If X < 0 ???
%     else
%         Y = X;
%     end
end
