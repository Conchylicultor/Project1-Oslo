function [ beta ] = IRLS( y, tX )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    % parametres
    maxIters = 10;
    beta = zeros(length(tX(1,:)),1); 
    
    for k = 1:maxIters     
        sig = sigmoid(tX*beta);
        s = sig.*(1-sig);
        z = tX*beta + (y-sig)./s;
        beta = weightedLeastSquares(z,tX,s);

        disp(['Iter:', num2str(costClass(y,tX,beta))]);
    end

end

function [ beta ] = weightedLeastSquares( z, tX, s )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    %Xt X z
    S = diag(s);
    M1 = tX'*S*z; 
    % XtsX
    M2 = tX'*S*tX;
    
    %M2^-1 M1
    beta = M2 \ M1;
    
end

