function [objective,rnvec,calls] = CalObj(Task,rnvec,p_il,options,i,skill_factor)
% ������Ⱥ���и����ڵ�i�������Ŀ�꺯��ֵ��factorial_costs��
% Input:
% Task������Ϣ��ά�ȣ����½磬��������rnvec�����۵�Ⱦɫ�塢p_ilԤѧϰ�ĸ��ʡ�optionsԤѧϰ��ѧϰ����i���۵�i������skill_factor������Ⱦɫ��ļ�������
% Output:
% objective������Ⱦɫ���Ŀ�꺯��ֵ��rnvec����Ԥѧϰ���磬�޸ĺ��Ⱦɫ�塢calls���۴���
    global N
    calls = 0;%���۴�����ʼ��Ϊ0
    calind = find(skill_factor == 0 | skill_factor == i);%�ҵ���Ҫ���۵���Ⱥ����
    objective = inf*ones(N,1);%Ŀ�꺯��ֵ��ʼ��
    d = Task.Tdims(i);
    pop = length(calind);
    objective1 = inf*ones(pop,1);%��Ҫ�����Ŀ�꺯��ֵ��ʼ��
    nvars = rnvec(calind,1:d);
    minrange = Task.Lb(i,1:d);
    maxrange = Task.Ub(i,1:d);
    y=repmat(maxrange-minrange,[pop,1]);
    vars = y.*nvars + repmat(minrange,[pop,1]);%����
    x = zeros(size(vars));%��ʱ����
    if rand(1)<=p_il
        for j = 1:pop
            [x(j,:),objective1(j),~,output] = fminunc(Task.fun(i).fnc,vars(j,:),options);%������Ҫ�����ĸ���ʹ����ţ�ٷ�����Ԥѧϰ
            calls=calls + output.funcCount;
        end
        nvars= (x-repmat(minrange,[pop,1]))./y;%��һ��
        m_nvars=nvars;
        m_nvars(nvars<0)=0;
        m_nvars(nvars>1)=1;
        if ~isempty(m_nvars~=nvars)%�������
            nvars=m_nvars;
            x=y.*nvars + repmat(minrange,[pop,1]);%����
            objective1=Task.fun(i).fnc(x);
        end
        rnvec(calind,1:d)=nvars;
    else
        x=vars;
        objective1=Task.fun(i).fnc(x);
        calls = length(calind);%��Ӧ�����۴���
    end
    objective(calind)=objective1;
end