


clear all

path='C:\Users\Administrator\Desktop\s\a.xlsx';

[ndata, text, alldata] = xlsread(path);

feaName=alldata(:,1);

temp={};
for i=1:5    
    for j=i:6:length(feaName)
        temp=[temp;alldata(j,:)];        
    end    
    temp=[temp;cell(1,size(temp,2))];
end


combine=temp;
xlsFile=['c:\','temp11111111111','.xlsx'];
xlswrite(xlsFile, combine); % 将need_xml_info直接转换成xls格式

