function y = clampp(x,a,b)

[d1,d2] = size(x);
low = round(a*d1*d2);
high = round(b*d1*d2);

so = sort(x(:));

low = so(low);
high = so(high);


x(x>high) = high;
x(x<low) = low;
y = x;
end