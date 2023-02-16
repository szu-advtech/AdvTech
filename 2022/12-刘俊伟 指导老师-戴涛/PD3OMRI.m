function [x, objective, time] = PD3OMRI(y, mask, sense,level,varargin)
%{
This function solves the regularization problem 
    arg min_x = 0.5*|| Mx-y||_2^2 + ||Gamma W Ax ||_1
 
%}
%% set default parameters
maxiter = 50;
lambda = 0.0035;
upsense = false;
win = 3;
gpu = false;
verbose = false;
method = '3DTF';
slambda = 100;

%% check input
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i = 1:2:(length(varargin)-1)
    switch lower(varargin{i})
      case 'lambda'
        lambda = varargin{i+1};
      case 'slambda'
        slambda = varargin{i+1};
      case 'maxiter'
        maxiter = varargin{i+1};
      case 'upsense'
        upsense = varargin{i+1};
      case 'winsize'
        win = varargin{i+1};
      case 'gpu'
        gpu = varargin{i+1};
      case 'method'
        method = varargin{i+1};
      case 'verbose'
        verbose = varargin{i+1};
      otherwise
        error(['Unrecognized option: ''' varargin{i} '''']);       
    end
  end
end

%% choose algorithm and count time
begin_time = tic;
switch method
  case '3DTF1'
    x = PD3O3DTF(y, lambda, mask, sense, maxiter, level, win, upsense, slambda, gpu);
  case '3DTF2'
    x = PD3O3DTF1(y, lambda, mask, sense, maxiter, level, win, upsense, slambda, gpu);
  case '2DTF'
    x = PD3O2DTF(y, lambda, mask, sense, maxiter, level, win, gpu);
    
  case 'TV'
    
  otherwise
    error(['Unrecognized method: ''' method '''']);  
end
end_time = toc(begin_time);
time = end_time;

%%show PD3O reconstruct figure
x = abs(x);
f4=figure;
movegui(f4,[1150,650]);
imshow(x,[]);
% title('PD3O重建结果图');
% FilName = ['./LJWResult/Simulation_','PD3O','.png'];
FilName = ['./LJWResult/Simulation_','PD3O','in-vivo_file_6_sr_33%','.png'];
imwrite(x./(max(x(:))-min(x(:))),FilName);


% h=imrect;
% position = getPosition(h);
% position = [105,194,41,23];
% region = imcrop(x,position);
% region = imresize(region,8,'nearest');
% figure;
% imshow(region,[min(x(:)),max(x(:))]);
% title('a part of region in PD3O picture');

% [x,y]=ginput(4);
% x_min=min(x);
% x_max=max(x);
% y_min=min(y);
% y_max=max(y);
% width = x_max-x_min;
% height = y_max - y_min;
% position = [x_min,y_min,width,height];
% roi = imcrop(x,position);
% figure(4);
% imshow(roi,[]);
% title('a part of region in PD3O picture');


residual = mask.*FFT2_3D_N(sense.*x) - y;
residual = residual(:);

objective = norm(residual);
fprintf('----------------------------\n');
if verbose
%   fprintf('Finished the reconstruction!\nResults:\n');
%   fprintf('||M u - g ||_2 = %10.3e\n',objective);
  if gpu
    fprintf('gpu time so far = %10.3e\n', time);
  else
    fprintf('PD3O算法运行时间为： %.5f秒\n', time);
  end
end
fprintf('----------------------------\n');

end

