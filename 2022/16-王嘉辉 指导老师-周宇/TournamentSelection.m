function index = TournamentSelection(K,N,varargin)
    varargin    = cellfun(@(S)reshape(S,[],1),varargin,'UniformOutput',false);
    [Fit,~,Loc] = unique([varargin{:}],'rows');
    [~,rank]    = sortrows(Fit);
    [~,rank]    = sort(rank);
    Parents     = randi(length(varargin{1}),K,N);
    [~,best]    = min(rank(Loc(Parents)),[],1);
    index       = Parents(best+(0:N-1)*K);
end