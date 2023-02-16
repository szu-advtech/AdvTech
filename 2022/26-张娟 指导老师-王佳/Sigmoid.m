x=-2:0.1:2;
N=length(x);

Y1 = zeros(1,N);
w = 1; q = 1.5; k=3;
for n = 1:N
      Y1(n) = 1/(1+exp(-k*(w*x(n)-q)));
end

Y2 = zeros(1,N);
w = -1; q = 0.5; k = 3;
for n = 1:N
    Y2(n) = 1/(1+exp(-k*(w*x(n)-q)));
end
subplot(2,1,1); plot(x,Y1);% 创建多个
subplot(2,1,2); plot(x,Y2)
