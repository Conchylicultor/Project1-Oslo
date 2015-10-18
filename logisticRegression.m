function [ beta ] = logisticRegression( y,tX,alpha )
%logisticRegression Logistic regression using gradient descent or Newton's method.	
%   Remark: alpha is the step size, in case of gradient descent.

    % Parameters
    maxIters = 1000;

    % Initialization
    beta = zeros(length(tX(1,:)), 1);

    % Main loop
    for k = 1:maxIters
        
        g = computeGradientLR(y, tX, beta);
        H = hessian(tX, beta);
        
        % Newtown method:
        beta = beta + alpha .* (H \ g); % Update beta

        % For debugging
        %L = computeCostLR(y, tX, beta);
        %fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    end
    
end

function [ L ] = computeCostLR(y, tX, beta )
%computeCostLR Compute the cost using Logistic Regression
    
    sum = 0;
    for i=1:length(y)
        yn = y(i);
        tXn = tX(i,:)';

        sum = sum + yn*tXn'*beta + log(1+tXn'*beta);    
    end

    L = -sum;
end

function [ g ] = computeGradientLR( y, tX, beta )
%computeGradientLR compute the gradient of a logistic regresstion step

    g = -(tX'*(fsigma(tX*beta) - y));

end
