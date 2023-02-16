function [cout] = project_to_F2(c)
% 将输入的反射系数c投影到F2上。  
% 不改变c的角度，将c的模化为1;

if c==0
    cout=1;
    return;
end

cout=c/norm(c);

end

