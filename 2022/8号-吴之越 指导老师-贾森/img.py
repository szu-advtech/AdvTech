# # from osgeo import gdal
# import scipy.io as scio
#
#
# if __name__ == '__main__':
#     dataFile = r'G:\A.mat'
#     dstpath = r'G:\B.tif'
#     dstpath = dstpath.encode('gbk', 'ignore')
#     data = scio.loadmat(dataFile)
#     if len(data['L'].shape) == 3: #L为在保存A.mat 前矩阵的名字
#         im_bands, im_height, im_width = data['L'].shape
#     else:
#         im_bands, (im_height, im_width) = 1, data['L'].shape
#
#     if 'int8' in data['L'].dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in data['L'].dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(dstpath, im_width, im_height, im_bands, datatype)
#
#     if im_bands == 1:
#         dataset.GetRasterBand(1).WriteArray(data['L'])  # 写入数组数据
#     else:
#         for i in range(im_bands):
#             dataset.GetRasterBand(i + 1).WriteArray(data['L'][i])
#
#     del dataset
