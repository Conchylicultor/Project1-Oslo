function [ beta ] = leastSquaresGD( y,tX,alpha )
%leastSquaresGD Least squares using gradient descent
%   Remark: alpha is the step-size

    % TODO: Implement stochastic gradient descent
    % QUESTION: Is least square the same as mean square error ???

    % Parameters
    maxIters = 1000;

    % Initialization
    beta = zeros(length(tX(1,:)), 1);

    % Main loop
    for k = 1:maxIters
        
        g = computeGradientMSE(y, tX, beta); % Compute gradient
        beta = beta - alpha .* g; % Update beta

        % For debugging
        % L = costMSE(y, tX, beta);
        % fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
    end

end

function [ g ] = computeGradientMSE( y, tX, beta )
%computeGradientMSE Compute the gradient for mean square error

    N = length(y);
    e = y - tX*beta;
    g = -1/N * tX'*e;

end
