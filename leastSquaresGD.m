function [ beta ] = leastSquaresGD( y,tX,alpha )
%leastSquaresGD Least squares using gradient descent
%   Remark: alpha is the step-size

    % TODO: Implement stochastic gradient descent

    % Parameters
    maxIters = 1000;

    % Initialization
    beta = zeros(length(tX(1,:)), 1);

    % Main loop
    for k = 1:maxIters
        
        g = computeGradientMSE(y, tX, beta); % Compute gradient
        beta = beta - alpha .* g; % Update beta

        % For debugging
        %L(k) = costMSE(y, tX, beta);
        %fprintf('%.2f  %.2f %.2f\n', L, beta(1), beta(2));
        
        if (g'*g < 1e-5) % Convergence (or local minimum)
            fprintf('Convergence');
            break; 
        end;
    end
    
    %figure(200)
    %plot(L);

end

function [ g ] = computeGradientMSE( y, tX, beta )
%computeGradientMSE Compute the gradient for mean square error

    N = length(y);
    e = y - tX*beta;
    g = -1/N * tX'*e;

end
