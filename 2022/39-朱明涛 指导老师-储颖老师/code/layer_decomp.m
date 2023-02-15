function [D1,D2,B2] = layer_decomp(img,lambda1,lambda2,lambda3)
    %% first scale
    [D1,B1] = L1L0(img,lambda1,lambda2);
    
    %% second scale
[row,col] = size(img);
shrink_scale = 4;
B1_low = imresize(B1,[round(row/shrink_scale),round(col/shrink_scale)],"bicubic");
[~,B2_low] = L1(B1_low,lambda3);
B2 = imresize(B2_low,[row,col],"bicubic");

%     [~,B2] = L1(B1,lambda3);

    D2 = B1 - B2;

end



