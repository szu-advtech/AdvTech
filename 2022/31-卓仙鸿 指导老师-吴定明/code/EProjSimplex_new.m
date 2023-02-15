% 出处 https://github.com/kunzhan/GSF
% 作者个人主页 https://github.com/kunzhan
function [x ft] = EProjSimplex_new(v, k)
% 解决simplex空间上的欧几里得投影问题
% 论文（11）式
% 也就是解决这个问题：

%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%  这是一个凸优化问题

if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end;
    end;
    x = max(v1,0);

else
    x = v0;
end;