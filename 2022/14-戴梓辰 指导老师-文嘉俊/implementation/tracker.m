function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);%��seqͼƬ���ж�ȡ��һ��ͼƬ��im
params = rmfield(params, 'seq');
if isempty(im)%�ж��Ƿ��ȡ����һ��ͼƬ
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);%����
    return;
end

% Init position
pos = seq.init_pos(:)';%��ʼ��Ŀ�������λ��
target_sz = seq.init_sz(:)';%Ŀ���ʵ�ʸߺͿ�
params.init_sz = target_sz;%��ʼ��Ŀ��ĸߺͿ�

% Feature settings
features = params.t_features;%��������������ȡ�ĺ�����get_colorspace,get_fhog,get_table_feature

% Set default parameters
params = init_default_params(params);%��ʼ�����ֲ���

% Global feature parameters
if isfield(params, 't_global')%����ȫ�ֲ���
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu%�ж��Ƿ�ʹ��gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');%������������
end
params.data_type_complex = complex(params.data_type);%��������Ϊ������ʽ

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;%���õ�����������һ��Ϊ2
init_penalty_factor = params.init_penalty_factor;%��ʼ��
max_penalty_factor = params.max_penalty_factor;%���æ�max
penalty_scale_step = params.penalty_scale_step;%���æ�
temporal_regularization_factor = params.temporal_regularization_factor; %����ʱ����������

init_target_sz = target_sz;

% Check if color image     %�ж��Ƿ�ΪRGB
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;%�ǲ�ɫͼ��
end

if size(im,3) > 1 && is_color_image == false%���������ͨ�� ������ͨ����� ���ж�Ϊ�ǲ�ɫͼ��
    im = im(:,:,1);%������ͨ����ȣ���ֻ��Ҫ��һ��ͨ��
end

% Check if mexResize is available and show warning otherwise.
% %ʹ��mexResize����
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try%�ж�mexResize�����Ƿ����
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);%search_area_scale=5 ��search_areaΪԭʼĿ���С���屶��
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;%�Գ�ʼ��Ŀ���������resize��ʹ��22500<���������С<40000

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;%200x200
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');%50x50
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');%4
num_feature_blocks = size(feature_sz, 1);%��������ʹ��ͬһ�������ߴ磬����features_blocks������Ϊ1

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);%��ȡimage_sample_size��image_input_size

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;%��֤�˲���filter_sz�Ĵ�С���������ߴ�Ĵ�С50x50
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);%�ҵ������˲�����С
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;%��Ϊifeatures_blocks������Ϊ1�����Ե���1
block_inds(k1) = [];

% Construct the Gaussian label function
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y); 
end

% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

% Define spatial regularization windows
reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};    %���������Ӧ�˲�����
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)%numel(reg_scale)=2
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end
params.reg_window=reg_window;
% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Define the learning variables
f_pre_f = cell(num_feature_blocks, 1);
cf_f = cell(num_feature_blocks, 1);
f_pre_f_pre = cell(num_feature_blocks, 1);

L = cell(num_feature_blocks, 1);
% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame >0
        [seq, im] = get_sequence_frame(seq);%frame=frame+1,�ٶ����֡ͼ��
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    % just want to initialize the tracker there
    if seq.frame > 1%������ǵ�һ֡
        old_pos = inf(size(pos));%inf�����������-inf������С��
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);%��������
            sample_scale = currentScaleFactor*scaleFactors;%���ж�߶ȱ任
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);%��ȡ5���߶��µ�����
                                    
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);%����ȡ�������������ϼ�cos window
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);%ת����Ƶ��
                        
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks. %��Ƶ���ж�ÿ��ͨ�����о�����㲢���
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds %block_indsΪ��
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
             
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);%socre_fs_sum=50x50x1x5,score_fs=50x50x5x1������5����5���߶ȣ�50x50����ÿ���߶��ϵ���Ӧ
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);%ͳһ�����߶�
            response = ifft2(responsef_padded, 'symmetric');%ת����ʵ��
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);
            
            %�ݶ��½����󼫴�ֵ�����Ծ�ȷ�������ؼ���
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(sind);            
            scale_change_factor = scaleFactors(sind);
            
            % update position
            old_pos = pos;%��¼��֡λ��
            pos = sample_pos + translation_vec;% translation_vec���������һ֡λ�õ�ƫ��
%             if sum(isnan(translation_vec))
%                 pos = sample_pos;
%             else
%                 pos = sample_pos + translation_vec;
%             end
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
                        
            % Update the scale
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extract image region for training sample
    sample_pos = round(pos);%�������룬pos�ǵ�ǰ֡��λ��
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);

    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);%��ȡ��������м�cos window����

    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);

    % train the CF model for each feature
%     for k = 1: numel(xlf)
%         model_xf = xlf{k};%Vj(x^)
% 
%         if (seq.frame == 1)
%             f_pre_f{k} = zeros(size(model_xf));%Ƶ���ǩy^��(Vj(f^t-1))
%             
%             mu = 0;
%         else
%             mu = temporal_regularization_factor(k);
%         end
%         
%         % intialize the variables
%         f_f = single(zeros(size(model_xf)));%  Vj(f^)
%         g_f = f_f;%g^
%         h_f = f_f;%h^
%         gamma  = init_penalty_factor(k);%1   1��gamma=1 (��)
%         gamma_max = max_penalty_factor(k);%0.1  0.1 ��gamma_max=0.1 (��max)
%         gamma_scale_step = penalty_scale_step(k);%10   10��gamma_scale_step=10  (��)
%         
%         % use the GPU mode
%         if params.use_gpu
%             model_xf = gpuArray(model_xf);
%             f_f = gpuArray(f_f);
%             f_pre_f{k} = gpuArray(f_pre_f{k});
%             g_f = gpuArray(g_f);
%             h_f = gpuArray(h_f);
%             reg_window{k} = gpuArray(reg_window{k});
%             yf{k} = gpuArray(yf{k});
%         end
% 
%         % pre-compute the variables
%         T = prod(output_sz);
%         S_xx = sum(conj(model_xf) .* model_xf, 3);%42��ͨ����ͺϲ�Ϊһ��ͨ��S_xx,   Vj(x^)(Vj(x^)ת)
%         Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);%ͬ��,  (Vj(x^)ת)Vj(f^t-1)
%         Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);% (Vj(x^)ת)Vj(f^t-1)
% 
%         % solve via ADMM algorithm
%         iter = 1;
%         while (iter <= admm_max_iterations)%����������
% 
%             % subproblem f
%             B = S_xx + T* (gamma + mu);%����Ӧ���� B = S_xx +  (gamma + mu) ��Eqn��9���еķ�ĸ����
%             Sgx_f = sum(conj(model_xf) .* g_f, 3);
%             Shx_f = sum(conj(model_xf) .* h_f, 3);
%  
%             f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - (( 1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
%                 bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
%                 (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);
%             %����Ӧ�����£�
% %             f_f = ((1/((gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((gamma/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
% %                 bsxfun(@rdivide,(1/((gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
% %                 (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);
% 
%             %   subproblem g
% %             g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));
%             
%             %����Ӧ��
%             g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ gamma*h_f)), g_f));
%             
%             
%             
% 
%             %   update h
% %             h_f = h_f + (gamma * (f_f - g_f));
%             
%             % ����Ӧ����
%             h_f = h_f + f_f - g_f;
% 
%             %   update gamma
%             gamma = min(gamma_scale_step * gamma, gamma_max);
%             
%             iter = iter+1;
%         end
%         
%         % save the trained filters
%         f_pre_f{k} = f_f;%(f^t-1)=(f^t)
%         
%        cf_f{k} = f_f;
%          
% %         if(seq.frame==1)
% %             cf_f{k}=f_f;
% %         else
% %             cf_f{k}=0.8*cf_f{k}+0.2*f_f;
% %         end
%     end  

    [cf_f, f_pre_f,f_pre_f_pre, L] = train_filter(cf_f, xlf, yf, f_pre_f,f_pre_f_pre, params, output_sz, seq);
            
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
                    
        drawnow
    end
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])

