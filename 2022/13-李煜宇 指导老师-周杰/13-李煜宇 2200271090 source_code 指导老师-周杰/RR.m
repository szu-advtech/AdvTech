function P_RR = RR(X, Y)
    P_RR = (X*X'+0.0001*eye(size(X*X')))\X*Y';
end