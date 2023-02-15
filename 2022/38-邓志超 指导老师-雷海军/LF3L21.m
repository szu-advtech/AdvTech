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
%        W: dxc, regression coefficient �ع�ϵ��
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
[fea,ins] = size(X); % ins ����������,fea ��������Ŀ
class = size(Y,1);

H_n = ins*eye(ins) - ones(ins); % ����ȡ��Ϊ��Ȩ����
H_c = class*eye(class) - ones(class);


% ���� X:93*92 ��ʾ92��������93��������. XXt:93*93 ��ʾ��������������93����(�����Լ�)�����.
% ���� XXt(2,4)����ʾ ��2�������4���������.
% XXt ��һ���Գƾ���.
XXt = X*X';
XHnXt = X*H_n*X'; %  �ȶ�X����һ����Ȩ������ע��:���ڸ���.

XYt = X*Y'; %  93��������Y�и��е����ϵ��.���ڸ���.
XHnYt = X*H_n*Y'; % �ȶ�X����һ����Ȩ���������ڸ�����93��������Y�и��е����ϵ��.
% ����д�� X * (Y' * H_c) ��������. 
XYtHc = X*Y'*H_c; % �������Ƕ�Y����һ����Ȩ������Ҳ��93��������Y�и��е����ϵ��.

b = 2*lambda2*H_c; 
% �����ͷ���ֵ��ת���� eps(0)
b(find(isinf(b))) = eps; % eps�Ǹ����͵ľ��ȣ�matlab��ʾΪ0
b(find(isnan(b))) = eps;


% �ȶ�XXt���� -A\B----�ȶ�A���棬Ȼ��*B(ע�������Ǿ���˷�),A\B=inv(A)*B   A/B=A*(inv(B))  
c = -XXt\(XYt+2*lambda1*XHnYt + 2*lambda2*XYtHc); % 93������������أ��������93������Y�и��е����.�������
c(find(isinf(c))) = eps; 
c(find(isnan(c))) = eps;
clear XYt XHnYt XYtHc

d = ones(fea,1); % fea ������ά��,����һ�£�����Ϊ93

for iter = 1:Ite % Ite ��������
    D = diag(d); % ÿ�ε���d����ͬ
    a = XXt\(XXt + 2*lambda1*XHnXt + lambda3*D); % ע�� a��c ������
    a(find(isinf(a))) = eps;
    a(find(isnan(a))) = eps;
    % a:93*93 b:3*3 c:93*3
    W = lyap(a,b,c); % ������ŵ�򷽳�-Sylvester equation �����㷨 AX+XB+C=0 �Ľ�,����A,BΪ����
    W21 = sqrt(sum(W.*W,2)) + eps; % sum(*,2)�Ƕ������
    d = 0.5./W21;  % Ҳ����д�� 1./(2*W21);  
    %obj(iter) = fValue(X,Y,lambda1,lambda2,lambda3,W,W21,H_n,H_c);   
end
%plot(obj)
end


% �������������������
function val = fValue(X,Y,lambda1,lambda2,lambda3,W,W21,H_n,H_c)
    val = norm(Y-W'*X,'fro')^2 + lambda1* trace(2*W'*X*H_n*X'*W - 4*Y*H_n*X'*W) + lambda2*trace(2*X'*W*H_c*W'*X - 4*X'*W*H_c*Y) + lambda3*sum(W21);
end


