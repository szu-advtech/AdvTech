%%% lab色彩空间转换为rgb色彩空间

function rgb = lab_to_rgb(lab)
cform = makecform('lab2srgb');
rgb = applycform(lab,cform);