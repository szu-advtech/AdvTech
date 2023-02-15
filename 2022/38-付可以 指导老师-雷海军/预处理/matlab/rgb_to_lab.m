%%% rgb色彩空间转换为lab色彩空间

function lab = rgb_to_lab(rgb)
cform = makecform('srgb2lab');
lab = applycform(rgb,cform);