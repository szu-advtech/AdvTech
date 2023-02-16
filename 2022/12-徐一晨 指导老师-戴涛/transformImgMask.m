% 对于sample-imageinpainting-HiFill的代码
% 缺失像素使用0表示，有效像素使用255表示，没有alpha层，mask一共有3个通道

% 此脚本可以将我使用的mask图像转存为对应的格式

clc;
clear;
srcDir = '../data/';		% 源文件存放位置
resImgDir = './samples/testset/';	% 结果文件存放位置
resMaskDir = './samples/maskset/';

files = dir(srcDir);
% 3:end 去除了'.'与'..'
files = files(end-1:end);

% 如果requiredMaxDim不等于0，且图像最大边长大于requiredMaxDim，则进行缩放
requiredMaxDim = 0;

resizeMethod = 'bicubic'; % nearest, bilinear, bicubic

for ii = 1:2:numel(files)
    % 读取图片和mask
    imgName = files(ii).name;
    maskName = files(ii+1).name;
    img = imread(fullfile(srcDir, imgName));
    mask = imread(fullfile(srcDir, maskName));
    
    imgH = size(img, 1);
    imgW = size(img, 2);
        
    if requiredMaxDim ~= 0 && max(imgH, imgW) > requiredMaxDim
        
        if imgH >= imgW
            newH = requiredMaxDim;
            newW = round(requiredMaxDim/imgH * imgW);
        else
            newW = requiredMaxDim;
            newH = round(requiredMaxDim/imgW * imgH);
        end
        
        img = sc_init_coarsest_level(img, ~mask);
        img = imresize(img, [newH, newW], resizeMethod);
        mask = imresize(mask, [newH, newW], resizeMethod);
        mask = mask > 0.5;
        
        img = img .* uint8(mask);
    end
    
    mask = repmat(uint8(~mask)*255, 1, 1, 3);
    
    % 保存结果
    imwrite(img, fullfile(resImgDir, imgName));
    imwrite(mask, fullfile(resMaskDir, maskName));
    
end

%%
% 对缺失区域进行拓扑填充，防止下采样时，缺失区域边界处产生edge
% reference: https://github.com/jbhuang0604/StructCompletion
function img = sc_init_coarsest_level(img, mask)

    % Get the inital solution
    [~, idMap] = bwdist(~mask, 'euclidean');

    % Intepolate only in the interior to avoid dark values near the image borders
    maskInt = mask;
    maskInt(1,:) = 0;   maskInt(end,:) = 0;
    maskInt(:,1) = 0;   maskInt(:,end) = 0;

    for ch = 1: 3
        imgCh = img(:,:,ch);
        imgCh = imgCh(idMap);
        img(:,:,ch) = regionfill(imgCh, maskInt);
    end

end