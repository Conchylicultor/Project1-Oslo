function [ beta ] = leastSquares( y,tX )
%leastSquares Least squares using normal equations.

    beta = (tX'*tX)\(tX'*y); %
    
end

