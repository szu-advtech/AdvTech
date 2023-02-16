import jittor as jt
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        input_dims = self.kwargs['input_dims']
        if self.kwargs['include_input']:
            embed_fns.append(lambda x:x)
            out_dim += input_dims
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        # freq_bands = 2. ** jt.linspace(0., max_freq, N_freqs)
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** jt.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, N_freqs)
            
        # print("freq_bands: ", freq_bands)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += input_dims
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return jt.contrib.concat([fn(inputs) for fn in self.embed_fns], -1)
        

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [jt.sin, jt.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim