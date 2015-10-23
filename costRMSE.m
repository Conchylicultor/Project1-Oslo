function [ L ] = costRMSE( y, tX, beta )
%costRMSE Root Mean Squared Error

    L = sqrt(2*costMSE(y, tX, beta)); % Multiply by 2 to compensate the divide by 2 in costMSE

end

