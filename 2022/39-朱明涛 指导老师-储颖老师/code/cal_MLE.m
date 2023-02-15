function [MLE_value] = cal_MLE(img)
    [row,col] = size(img);
    MLE_value = 0;
    N = 0;
    i = 1;
    while i<=(row-8)
        j = 1;
        while j<=(col-8)
            localimg = img(i:i+8,j:j+8);
            LE = 0;
            for m=1:9
                for n=1:9
                    Pi = sum(localimg(:)==localimg(m,n))/81;
                    LE = -Pi*log10(Pi) + LE;
                end
            end
            MLE_value = MLE_value + LE;
            N = N + 1;
            j = j+9;
        end
        i = i + 9;
    end
    MLE_value = MLE_value/N;
end

