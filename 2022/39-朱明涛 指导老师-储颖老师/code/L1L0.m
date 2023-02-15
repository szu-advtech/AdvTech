function [D1,B1] = L1L0(S,lambda1,lambda2)
    [row,col] = size(S);
    iter = 15;

    fx = [1,-1];
    fy = [1;-1]; % 梯度算子
    fftGfx = psf2otf(fx,[row,col]);  % fft(▽x)
    fftGfy = psf2otf(fy,[row,col]);  % fft(▽y)
%     DxDy = abs(fftGfy).^2 + abs(fftGfx).^2;
    B = S;
    C1 = zeros(row,2*col);
    C2 = zeros(row,2*col);
    Y1 = zeros(row,2*col);
    Y2 = zeros(row,2*col);

    ro = 1;
    GradS = [-imfilter(S,fx,'circular'),-imfilter(S,fy,'circular')];  % ▽S
    
    for i=1:iter
        C1_ = C1 + Y1./ro;
        C2_ = GradS - C2 - Y2./ro;
        
        %% 求B
        C1_1 = C1_(:,1:col);
        C1_2 = C1_(:,col+1:end);
        C2_1 = C2_(:,1:col);
        C2_2 = C2_(:,col+1:end);
        
        Numet = fft2(S) + ro.*conj(fftGfx).*(fft2(C1_1)+fft2(C2_1)) + ro.*conj(fftGfy).*(fft2(C1_2)+fft2(C2_2));
        Dnomt = 1 + 2*ro.*(conj(fftGfx).*fftGfx+conj(fftGfy).*fftGfy);
%         Dnomt = 1 + 2*ro.*DxDy;
        B_new = real(ifft2(Numet./Dnomt));
        GradB = [-imfilter(B_new,fx,'circular'),-imfilter(B_new,fy,'circular')];  % ▽B

        %% 求C1
        X = GradB - Y1./ro;
        C1_new = sign(X).*max(abs(X)-lambda1./ro,0);

        %% 求C2
        Qk = GradS - GradB - Y2./ro;
        C2_new = Qk;
        condition = Qk.^2 < 2*lambda2./ro;
        C2_new(condition) = 0;

        %% 求Y1，Y2
        Y1_new = Y1 + ro.*(C1_new-GradB);
        Y2_new = Y2 + ro.*(C2_new-GradS+GradB);

        %% ro
        ro = ro*2;

        %% 更新
        B = B_new;
        C1 = C1_new;
        C2 = C2_new;
        Y1 = Y1_new;
        Y2 = Y2_new;
    end
    B1 = B;
    D1 = S-B;

end

