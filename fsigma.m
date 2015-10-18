function [y] = fsigma(x)
%fsigma Compute sigma(x)
    y = 1 ./ (1+exp(-x));
end
