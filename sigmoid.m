function [y] = sigmoid(x)
%sigmoid Compute sigma(x)
    y = 1 ./ (1+exp(-x));
end
