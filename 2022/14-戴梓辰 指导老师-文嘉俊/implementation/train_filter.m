function [cf_f, f_pre_f,f_pre_f_pre, L] = train_filter(cf_f, xlf, yf, ...
    f_pre_f,f_pre_f_pre , params, output_sz, seq)
    for k = 1: numel(xlf)%只循环一次
        model_xf = xlf{k};%Vj(x^)

        if (seq.frame == 1)
            f_pre_f{k} = zeros(size(model_xf));%频域标签y^，(Vj(f^t-1))
            f_pre_f_pre{k}=f_pre_f{k};
            mu = 0;
        else
            mu = params.temporal_regularization_factor(k);
        end
        
        % intialize the variables
        f_f = single(zeros(size(model_xf)));%  Vj(f^)
        g_f = f_f;%g^
        h_f = f_f;%h^
        gamma  = params.init_penalty_factor(k);%1   1，gamma=1 (γ)
        gamma_max = params.max_penalty_factor(k);%0.1  0.1 ，gamma_max=0.1 (γmax)
        gamma_scale_step = params.penalty_scale_step(k);%10   10，gamma_scale_step=10  (ρ)
        
        % use the GPU mode
%         if params.use_gpu
%             model_xf = gpuArray(model_xf);
%             f_f = gpuArray(f_f);
%             f_pre_f{k} = gpuArray(f_pre_f{k});
%             g_f = gpuArray(g_f);
%             h_f = gpuArray(h_f);
%             reg_window{k} = gpuArray(reg_window{k});
%             yf{k} = gpuArray(yf{k});
%         end

        % pre-compute the variables
        T = prod(output_sz);
        S_xx = sum(conj(model_xf) .* model_xf, 3);%42个通道求和合并为一个通道S_xx,   Vj(x^)(Vj(x^)转)
        Sf_pre_f = sum(conj(model_xf) .* f_pre_f{k}, 3);%同上,  (Vj(x^)转)Vj(f^t-1)
        Sfx_pre_f = bsxfun(@times, model_xf, Sf_pre_f);% (Vj(x^)转)Vj(f^t-1)

        % solve via ADMM algorithm
        iter = 1;
        while (iter <= params.max_iterations)%最大迭代两次

            % subproblem f
            B = S_xx + T* (gamma + mu);%文中应该是 B = S_xx +  (gamma + mu) 即Eqn（9）中的分母部分
            Sgx_f = sum(conj(model_xf) .* g_f, 3);
            Shx_f = sum(conj(model_xf) .* h_f, 3);
 
            f_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - (( 1/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (0.9*mu/(gamma + mu)) * f_pre_f{k}+(0.1*mu/(gamma + mu)) * f_pre_f_pre{k} - ...
                bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
                (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);
            %文中应是如下：
%             f_f = ((1/((gamma + mu)) * bsxfun(@times,  yf{k}, model_xf)) - ((gamma/(gamma + mu)) * h_f) +(gamma/(gamma + mu)) * g_f) + (mu/(gamma + mu)) * f_pre_f{k} - ...
%                 bsxfun(@rdivide,(1/((gamma + mu)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (mu/(gamma + mu)) * Sfx_pre_f - ...
%                 (1/(gamma + mu))* (bsxfun(@times, model_xf, Shx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, model_xf, Sgx_f))), B);

            %   subproblem g
%             g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ h_f)), g_f));
            
            %文中应是
%             g_f = fft2(argmin_g(reg_window{k}, gamma, real(ifft2(gamma * f_f+ gamma*h_f)), g_f));
      
            X = real(ifft2(gamma * f_f+ h_f));
            if (seq.frame == 1)
                X_temp = zeros(size(X));     
                for i = 1:size(X,3)
                 X_temp(:,:,i) = X(:,:,i) ./  (params.reg_window{k} .^2 + gamma);
                end
                L = 0;
            else  
        
            X_temp=X;
            L{k} = max(0,1-1./(gamma*numel(X)*sqrt(sum(X_temp.^2,3))));
    
            [~,b] = sort(L{k}(:),'descend');
            L{k}(b(ceil(params.fs_rate(k)*1/params.search_area_scale^2*numel(b)):end)) = 0;
    
            X_temp = repmat(L{k},1,1,size(X_temp,3)) .* X_temp;
            end
            
            g_f = fft2(X_temp);
            

            %   update h
            h_f = h_f + (gamma * (f_f - g_f));
            
            % 文中应该是
%             h_f = h_f + f_f - g_f;

            %   update gamma
            gamma = min(gamma_scale_step * gamma, gamma_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        f_pre_f_pre{k}=f_pre_f{k};
        f_pre_f{k} = f_f;%(f^t-1)=(f^t)
        
%         cf_f{k} = f_f;
         
        if(seq.frame==1)
            cf_f{k}=f_f;
        else
            cf_f{k}=0.95*cf_f{k}+0.05*f_f;
        end
    end
end