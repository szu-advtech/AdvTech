function [D2,B2] = L1(S,lambda3)
    iter = 10;
    [row,col] = size(S);
    
    fx = [1, -1];
    fy = [1; -1];
    fftGfx = psf2otf(fx,[row,col]);
    fftGfy = psf2otf(fy,[row,col]);
    
    B = S; % b
    C1 = zeros(row,col); % c11
    C2 = zeros(row,col); % c12
    Y1 = zeros(row,col); % y11
    Y2 = zeros(row,col); % y12
    ro = 1;

    for i = 1:iter
        
        %% 求B
        Nomi = fft2(S) + ro * conj(fftGfx) .* fft2(C1 + Y1./ro) + ro * conj(fftGfy) .* fft2(C2 + Y2./ro);
        Denomi = 1 + ro * (conj(fftGfx).*fftGfx+conj(fftGfy).*fftGfy);
        B_new = real(ifft2(Nomi./Denomi));
        
        GradxB = -imfilter(B_new,fx,'circular'); % ▽x B
        GradyB = -imfilter(B_new,fy,'circular'); % ▽y B
        %% 求C1,C2
        BB1 = GradxB - Y1./ro;
        BB2 = GradyB - Y2./ro;
        C1_new = sign(BB1) .* max(abs(BB1) - lambda3.*1./ro ,0);
        C2_new = sign(BB2) .* max(abs(BB2) - lambda3.*1./ro ,0);
        
        %% 求Y1, Y2
        Y1_new = Y1 + ro * (C1_new - GradxB);    
        Y2_new = Y2 + ro * (C2_new - GradyB);
        
        %% ro
        ro = ro * 2;
        
        %% 更新
        B = B_new;
        C1 = C1_new;
        C2 = C2_new;
        Y1 = Y1_new;
        Y2 = Y2_new;
    end
    B2 = B;
    D2 = S - B;
end

