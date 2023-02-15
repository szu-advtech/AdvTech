function [ACCIndex lableRef] = ACC2(gnd,lable,C)
%%%%%%%%gnd 标准分类；lable:求得的结果；C 类数%%%%%%%%%%%%%
  
 lableRef = bestMap(gnd,lable);

 ACCIndex = sum(lableRef(:) == gnd(:)) / length(gnd);