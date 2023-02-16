#!/bin/bash
# 使用此脚本将result文件夹中的图像修复结果图片进行重命名
# 使用方法: sh rename-result.sh
# windows下使用git bash运行sh文件

imgDir='./result'
imgNum=86

# 遍历imgDir中的所有文件
for file in `ls $imgDir`
do
	origPath=$(printf '%s/%s' "$imgDir" "$file")
	newPath=$(printf '%s/%03d_high_fill.png' "$imgDir" "$imgNum")
	printf 'rename %s -> %s\n' "$origPath" "$newPath"
	mv $origPath $newPath
	
	imgNum=$(($imgNum+1))
done