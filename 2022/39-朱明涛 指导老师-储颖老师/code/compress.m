function y = compress(x,gamma,W)
% W使最大的亮度级别，由于归一化，一般为1

if nargin<3
    W = 1;
end
    
y = W * ((x./W) .^ (1./gamma));

end