function [W, obj]= LF3L21(X,Y,lambda1,lambda2,lambda3,Ite)

% objective function: 
%              min_W |Y-W^TX|_F^2 + lambda1*\sum_ji^n((y_i-y_j)-(W^Tx_i-W^Tx_j))^2 + lambda2*\sum_pq^c((y_p^T-y_q^T)-(X^Tw_p-X^Tw_q))^2+ lambda3*2|W|_2,1.
%    Matrix form: 
%              min_W |Y-W^TX|_F^2 + lambda1* tr(2W^TXH_nX^TW - 4YH_nX^TW) + lambda2* (2tr(X^TWH_cW^TX) - 4tr(X^TWH_cY))+ lambda3*2|W|_2,1.
%    solution: 
%                 1: XX^T^{-1}(XX^T + 2*lambda1*X*Hn*X' + lambda3*D)W + W(2lambda2*H_c) - XX^T\(XY^T + 2*lambda1*XY^T + 2lambda2*X*Y^T*Hc) = 0.
%                 2:lyap
%             or 1:(XX^T + 2*lambda1*X*Hn*X' + lambda3*D)W*H_c +2lambda2*XX^TW*H_c -(XY^T*H_c + 2*lambda1*XY^T*H_c + 2lambda2*X*Y^T*Hc) = 0.
%                2:(XX^T + 2*lambda1*X*Hn*X' + 2lambda2*XX^T + lambda3*D)W*H_c = (XY^T*H_c + 2*lambda1*XY^T*H_c + 2lambda2*X*Y^T*Hc)
%                3:W*H_c=(XX^T + 2*lambda1*X*Hn*X' + 2lambda2*XX^T +lambda3*D)\(XY^T*H_c + 2*lambda1*XY^T*H_c + 2lambda2*X*Y^T*Hc)
%                4.W = (W*Hc)/Hc
% input:
%        X: dxn, n is the number of instances, d is the number of dimensions, 
%        X: feature vectors, each row is an instance
%        Y: cxn, c is the number of class label,
%        lambda_i: tuning parameters
%        Ite: iteration number, e.g., 50
%        H_n,H_c: centering matrix, where H_n = eye(n) - 1/n*ones(n); %% ones(n) = 1_n1_n^T
% output:
%        W: dxc, regression coefficient 回归系数
% exmaple:  
%        clear;clc;[W obj]= LF3L21(rand(50,500),rand(20,500),1,1,1200,100);
%        clear;clc;load SVM_xf;[W obj]= LF3L21(X,Y,1,1,0.01,100);
%        1st edition by Xiaofeng Zhu 6/20/2013


% function [W, obj]= LF3L21(X,Y,lambda1,lambda2,lambda3,Ite)
if (~exist('lambda1','var'))    lambda1 = 1;    end
if (~exist('lambda2','var'))    lambda2 = 1;    end
if (~exist('lambda3','var'))    lambda3 = 50;   end % 50 ?
if (~exist('lambda4','var'))    lambda4 = 1;    end
if (~exist('Ite','var'))        Ite = 50;       end


% initial
[fea,ins] = size(X); % ins 是样本数量,fea 是特征数目
class = size(Y,1);

H_n = ins*eye(ins) - ones(ins); % 暂且取名为加权矩阵
H_c = class*eye(class) - ones(class);


% 假如 X:93*92 表示92个样本，93个脑区域. XXt:93*93 表示各个区域与其他93区域(包括自己)的相关.
% 例如 XXt(2,4)，表示 第2脑区与第4脑区的相关.
% XXt 是一个对称矩阵.
XXt = X*X';
XHnXt = X*H_n*X'; %  先对X进行一个加权操作，注意:存在负数.

XYt = X*Y'; %  93个脑区与Y中各行的相关系数.存在负数.
XHnYt = X*H_n*Y'; % 先对X进行一个加权操作，存在负数。93个脑区与Y中各行的相关系数.
% 可以写成 X * (Y' * H_c) 满足结合律. 
XYtHc = X*Y'*H_c; % 这里先是对Y进行一个加权操作。也是93个脑区与Y中各行的相关系数.

b = 2*lambda2*H_c; 
% 无穷大和非数值都转换成 eps(0)
b(find(isinf(b))) = eps; % eps是浮点型的精度，matlab显示为0
b(find(isnan(b))) = eps;


% 先对XXt求逆 -A\B----先对A求逆，然后*B(注意这里是矩阵乘法),A\B=inv(A)*B   A/B=A*(inv(B))  
c = -XXt\(XYt+2*lambda1*XHnYt + 2*lambda2*XYtHc); % 93个脑区的自相关，调整后的93脑区与Y中各行的相关.进行左除
c(find(isinf(c))) = eps; 
c(find(isnan(c))) = eps;
clear XYt XHnYt XYtHc

d = ones(fea,1); % fea 是特征维度,保持一致，假设为93

for iter = 1:Ite % Ite 迭代次数
    D = diag(d); % 每次迭代d都不同
    a = XXt\(XXt + 2*lambda1*XHnXt + lambda3*D); % 注意 a与c 的区别
    a(find(isinf(a))) = eps;
    a(find(isnan(a))) = eps;
    % a:93*93 b:3*3 c:93*3
    W = lyap(a,b,c); % 李雅普诺夫方程-Sylvester equation 分裂算法 AX+XB+C=0 的解,其中A,B为方阵
    W21 = sqrt(sum(W.*W,2)) + eps; % sum(*,2)是对行求和
    d = 0.5./W21;  % 也可以写成 1./(2*W21);  
    %obj(iter) = fValue(X,Y,lambda1,lambda2,lambda3,W,W21,H_n,H_c);   
end
%plot(obj)
end


% 下面这个函数？？？？
function val = fValue(X,Y,lambda1,lambda2,lambda3,W,W21,H_n,H_c)
    val = norm(Y-W'*X,'fro')^2 + lambda1* trace(2*W'*X*H_n*X'*W - 4*Y*H_n*X'*W) + lambda2*trace(2*X'*W*H_c*W'*X - 4*X'*W*H_c*Y) + lambda3*sum(W21);
end


