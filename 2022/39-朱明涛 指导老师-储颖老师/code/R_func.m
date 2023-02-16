function y = R_func(x,g,sigma,a,b)

index_low = abs(x-g) <= sigma;
temp = x(index_low);
temp_low = g + fd( abs(temp - g)./sigma, a) .* sigma .* sign(temp - g);
% temp_low = g + betainc( abs(temp - g)./sigma,a*2,a*4) .* sigma .* sign(temp - g);

index_high = abs(x-g) > sigma;
temp = x(index_high);
temp_high = g + (fe( abs(temp - g) - sigma, b) + sigma) .* sign(temp - g);


y = zeros(size(x));
y(index_low) = temp_low;
y(index_high) = temp_high;


function y = fd(x,a)
    y = x.^a;
end

function y = fe(x,b)
    y = b*x;
end

end