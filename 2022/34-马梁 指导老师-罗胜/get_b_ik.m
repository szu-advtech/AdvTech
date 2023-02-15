function [b_ik] = get_b_ik(i,k,w)
%GET_B_IK Summary of this function goes here
%   Detailed explanation goes here
load("generate_channel.mat","h_dk")
b_ik=h_dk(:,k)'*w(:,i);
end

