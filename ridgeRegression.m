function [ beta ] = ridgeRegression( y,tX, lambda )
%ridgeRegression Ridge regression using normal equations.
%   Remark: lambda is the regularization coefficient.

    I = eye(length(tX(1,:))); % Creation of an identity matrix of dimention (M+1)
    I(1,1) = 0;
    beta = (tX'*tX + lambda*I)\(tX'*y);
    
end

