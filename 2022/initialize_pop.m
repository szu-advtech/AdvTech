function [ POP ] = initialize_pop(n,c,bu,bd)
% Usage: [ POP ] = initialize_pop(n,c,bu,bd)
%
% Input:
% bu            -Upper Bound
% bd            -Lower Bound
% c             -No. of Decision Variables
% n             -Population Scale
%
% Output: 
% POP           -Initial Population
%------------------------------------------------------------------------
POP=lhsdesign(n,c).*(ones(n,1)*(bu-bd))+ones(n,1)*bd;

end