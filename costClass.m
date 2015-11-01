function [ L ] = costClass( y, tX, beta )
%costClass Compute how well our classification perform

classOnePrediction = sigmoid(tX*beta) > 0.5;
yPredicted = 1*classOnePrediction -1*~classOnePrediction;
L = sum(y ~= yPredicted) / length(y);

end

