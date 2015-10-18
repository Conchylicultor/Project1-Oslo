function [H] = hessian(tX, beta)
%hessian

    N = length(tX(:,1));
    S = zeros(N);
    for i=1:N
        tXn = tX(i,:)';
        S(i,i) = fsigma(tXn'*beta)*(1-fsigma(tXn'*beta));
    end
    H = tX'*S*tX;
end
