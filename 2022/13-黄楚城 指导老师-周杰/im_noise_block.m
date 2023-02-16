function A=im_noise_block(A,csize,type, density)
%add random corruption,A si and (m*n)  *training_number 2D image
%usually,csize=20% of the image size
% density for salt and pepper noise

[imline, imcolumn, numall]=size(A);
position=rand(imline, imcolumn);
%for i=1:classnum
%    for j=1:trainingnuminclass
%        count1=(classnum-1)*trainingnuminclass;
%        x=max(fix(rand(1)*imline),1);y=max(fix(rand(1)*imcolumn),1);
%        if x>=imline-csize x=imline-csize;end
%        if y>=imcolunm-csize y=imcolunm-csize;end
%        A(x:x+csize,y:y+csize,count)=A(x:x+csize,y:y+csize,count)/5;
 %   end
%end
if type==1                 %---------------block subtraction
    for i=1:numall
        x=max(fix(rand(1)*imline),1);y=max(fix(rand(1)*imcolumn),1);
        if x>=imline-csize x=imline-csize;end
        if y>=imcolumn-csize y=imcolumn-csize;end
        A(x:x+csize,y:y+csize,i)=0;
    end
end
if type==2                 %----------------noise only
    for i=1:numall
       A(:,:,i) = awgn(A(:,:,i),density, 'measured');     
    end
end
if type==3                 %---------------block subtraction with white noise
     for i=1:numall
        x=max(fix(rand(1)*imline),1);y=max(fix(rand(1)*imcolumn),1);
        if x>=imline-csize x=imline-csize;end
        if y>=imcolumn-csize y=imcolumn-csize;end
        A(x:x+csize,y:y+csize,i)=0;
        A(:,:,i) = awgn(A(:,:,i),1, 35); 
     end
end
if type==4                 %---------------block cover with random noise 
    for i=1:numall
        x=max(fix(rand(1)*imline),1);y=max(fix(rand(1)*imcolumn),1);
        if x>=imline-csize x=imline-csize;end
        if y>=imcolumn-csize y=imcolumn-csize;end
        A(x:x+csize,y:y+csize,i)=A(x,y)*abs(rand(csize+1,csize+1));
    end
end
if type==5                 %----------------noise only
    for i=1:numall
       A(:,:,i) = imnoise(A(:,:,i),'salt & pepper',density);     
    end
end

    