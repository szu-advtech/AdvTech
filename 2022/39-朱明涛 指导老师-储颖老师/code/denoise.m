function D = denoise(img,alp)
M = max(img(:));
[row,col] = size(img);
D = img;
M1 = alp*M;
M2 = (1-alp)*M;
for i=1:row
    for j=1:col
        if D(i,j)<M1||D(i,j)>M2
            D(i,j)=0;
        end
      
    end
end
end

