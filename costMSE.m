function [ L ] = costMSE( y, tX, beta )
%costMSE Compute the cost using mean square error

    N = length (y);
    e = y - tX*beta; % Compute error
    L = e'*e/(2*N); % Compute MSE
    
end

