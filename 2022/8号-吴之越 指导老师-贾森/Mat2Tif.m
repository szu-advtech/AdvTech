function Mat2Tif(InputMatFileName,OutputTifFilename)
    load(InputMatFileName);
    InputMatImg=uint32(chikusei);
    t = Tiff(OutputTifFilename,'w');
    if size(InputMatImg,3) == 3
        t.setTag('Photometric',Tiff.Photometric.RGB);
    else
        t.setTag('Photometric',Tiff.Photometric.MinIsBlack);%颜色空间解释方式
    end
    t.setTag('Compression',Tiff.Compression.None);%无压缩
    t.setTag('BitsPerSample',32);% 由于输入.mat为uint32类型，所以选择了32位
    t.setTag('SamplesPerPixel',size(InputMatImg,3));% 每个像素的波段数目
%     t.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP);% 配合BitsPerSample64位double类型，选择IEEEFP来对应
    t.setTag('ImageLength',size(InputMatImg,1));% 影像宽度
    t.setTag('ImageWidth',size(InputMatImg,2));% 影像高度
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);%平面配置选择集中式
    t.write(InputMatImg);% 准备好了头文件，开始写影像数据
    t.close();% 关闭影像
