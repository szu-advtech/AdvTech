%{
This code is the criteria of Multi-label learning.

Input:
    pre   : the predict label, n*c dimensions, each row is a label
    label : the real label, n*c dimensions, each row is a label

Output:
    criteria: a struct including
            hammloss    : Hamming loss
            rankloss    : Ranking loss
            one_er      : One-error
            Coverage_ac : Coverage
            avg_prec    : Average precision
    This criterias can be found in: 
https://www.geeksforgeeks.org/multilabel-ranking-metrics-ranking-loss-ml/?ref=lbp
%}

function [criteria] = CriteriaForMLL(pre, data)
    % Hamming loss
    hammloss = 0;
    for i = 1:size(data.label,1)
        if isempty(find(data.label(i,:) == 1,1))
            continue;
        end
        hammloss = hammloss + sum(xor(pre(i,:),data.label(i,:)))/data.infm.nr_class;
    end
    criteria.hammloss = hammloss / data.infm.nr_sample;

    % Ranking loss
    rankloss = 0;
    for i = 1:size(data.label,1)
        if isempty(find(data.label(i,:) == 1,1))
            continue;
        end
        [~,tmp] = sort(pre(i,:),'descend');
        tmp_count = 0;
        tmp_num = 0; % count the number of prediction WRONG
        for i1 = 1:data.infm.nr_class
            if (data.label(i,tmp(i1)) == 0)
                tmp_num = tmp_num + 1;
            else
                tmp_count = tmp_count + tmp_num;
            end
        end
        tmp_num = sum(data.label(i,:))*(data.infm.nr_class-sum(data.label(i,:)));
        rankloss = rankloss + tmp_count/tmp_num;
    end
    criteria.rankloss = rankloss / data.infm.nr_sample;

    % One-error
    one_er = 0;
    for i = 1:size(data.label,1)
        if isempty(find(data.label(i,:) == 1,1))
            continue;
        end
        [~,tmp] = sort(pre(i,:),'descend');
        if (data.label(i,tmp(1)) ~= 1)
            one_er = one_er + 1;
        end
    end
    criteria.one_er = one_er / data.infm.nr_sample;

    % Coverage
    Coverage_ac = 0;
    for i = 1:size(data.label,1)
        if ~isempty(find(data.label(i,:) == 1,1))
            [~,tmp] = sort(pre(i,:),'descend');
            tmp_al = find(data.label(i,:) == 1);
            tmp_max = find(tmp == tmp_al(1));
            for j = 2:length(tmp_al)
                if (find(tmp == tmp_al(j))>tmp_max)
                    tmp_max = find(tmp == tmp_al(j));
                end
            end
        else
            tmp_max = 1;
        end
        Coverage_ac = Coverage_ac + tmp_max - 1;
    end
    criteria.Coverage_ac = Coverage_ac / data.infm.nr_sample;

    % Average Precision
    avg_prec = 0;
    for i = 1:size(data.label,1)
        if isempty(find(data.label(i,:) == 1,1))
            continue;
        end
        [~,tmp] = sort(pre(i,:),'descend');
        tmp_count = 0;
        for i1 = find(data.label(i,:) == 1)
            tmp_rank = tmp(i1);
            tmp_L = 0;
            for i2 = 1:data.infm.nr_class
                if (data.label(i,tmp(i1)) == 1 && tmp(i1) < tmp(i2))
                    tmp_L = tmp_L + 1;
                end
            end
            tmp_count = tmp_count + tmp_L/tmp_rank;
        end
        avg_prec = avg_prec + tmp_count/sum(data.label(i,:));
    end
    criteria.avg_prec = avg_prec / data.infm.nr_sample;
end
