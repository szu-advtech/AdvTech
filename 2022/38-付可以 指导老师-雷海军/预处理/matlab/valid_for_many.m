file_dirname = ''
file_dir = dir(file_dirname);
file_dir(1:2) = [];

file_nums = length(file_dir);

for i = 1:file_nums
    image1 = im2double(imread(fullfile(file_dirname,file_dir(i).name)));
    
    %%% WBP——动态阈值自动白平衡算法
    hazy_wb = SimplestColorBalance(uint8(255*image1));
    hazy_wb = uint8(hazy_wb);
    
    %%% CLAHE——全局化限制对比度自适应直方图均衡
    lab1 = rgb_to_lab(uint8(255*image1));
    lab2 = lab1;
    lab2(:, :, 1) = adapthisteq(lab2(:, :, 1));
    img2 = lab_to_rgb(lab2);
    hazy_cont = img2;

    %%% 伽马校正
    hazy_gamma = image1.^0.7;
    hazy_gamma =  uint8(255*hazy_gamma);
    
    image1 = uint8(255*image1);
    
    imwrite(image1,fullfile('input_test',file_dir(i).name ));
    imwrite(hazy_wb,fullfile('input_wb_test',file_dir(i).name ));
    imwrite(hazy_cont,fullfile('input_ce_test',file_dir(i).name ));
    imwrite(hazy_gamma,fullfile('input_gc_test',file_dir(i).name ));

end