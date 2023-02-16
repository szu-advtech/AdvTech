function y  = normalize(x)
    y = (x-min(x(:)))/(max(x(:))-min(x(:)));

end