function [ beta ] = penLogisticRegression( y,tX,alpha,lambda )
%penLogisticRegression Penalized logistic regression using gradient descent or Newton's method.
%   Remark: alpha is the step size for gradient descent, lambda is the regularization parameter

    % Parameters
    maxIters = 1000;

    % Initialization
    beta = zeros(length(tX(1,:)), 1);

    % Main loop
    for k = 1:maxIters
        
        g = computeGradientPLR(y, tX, beta,lambda);
        H = hessian(tX, beta);
        
        % Newtown method:
        beta = beta + alpha .* (H \ g); % Update beta

        % For debugging
        %L = computeCostPLR(y, tX, beta,lambda);
        %fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    end
end

function [ L ] = computeCostPLR(y, tX, beta,lambda )
%computeCostLR Compute the cost using Logistic Regression
    
    sum = 0;
    for i=1:length(y)
        yn = y(i);
        tXn = tX(i,:)';

        sum = sum + yn*tXn'*beta + log(1+tXn'*beta);    
    end

    % Adding regularization term
    for i=1:length(beta)
        sum = (lambda/2)*beta(i)^2;    
    end

    L = -sum;
end

function [ g ] = computeGradientPLR( y, tX, beta, lambda )
%computeGradientLR compute the gradient of a logistic regresstion step
%   lambda is the regularisation term

    g = -(tX'*(sigmoid(tX*beta) - y) + lambda*beta);

end
