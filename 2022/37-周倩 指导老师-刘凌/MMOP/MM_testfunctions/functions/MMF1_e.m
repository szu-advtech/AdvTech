function y = MMF1_e(x)

% 1<=x1<=3    -1<=x2<=1
% x       number_of_point * number_of_decision_var
% y       number_of_point * number_of_objective 

    left_index=find(x(:,1)<2);
    right_index=find(x(:,1)>=2);
    y(left_index,1)      = 2-x(left_index,1);
    y(right_index,1)      = x(right_index,1)-2;
    y(left_index,2)=1.0 - sqrt(2-x(left_index,1)) + 2.0*(x(left_index,2)-sin(6*pi*(2-x(left_index,1))+pi)).^2;
    y(right_index,2)=1.0 - sqrt(x(right_index,1)-2) + 2.0*(x(right_index,2)-exp(x(right_index,1)).*sin(6*pi*(x(right_index,1)-2)+pi)).^2;
end


