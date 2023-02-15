import torch


def rigid_alignment(cf, sf, alpha=0, beta=0, s1f=None):
    
    # content image
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    zc = cf.view(c_channels, -1)
    
    mean_c = torch.mean(zc, 0)
    mean_c = mean_c.unsqueeze(0).expand_as(zc)
    zc_bar = zc - mean_c
    
    zc_cap = zc_bar/torch.norm(zc_bar)
    
    # style image
    sf = sf.double()
    s_channels, s_width, s_height = sf.size(0), sf.size(1), sf.size(2)
    zs = sf.view(s_channels, -1)
    
    mean_s = torch.mean(zs, 0)
    mean_s = mean_s.unsqueeze(0).expand_as(zs)
    zs_bar = zs - mean_s
    
    zs_cap = zs_bar/torch.norm(zs_bar)
    
    # rotation
    usvt = torch.mm(torch.transpose(zc, 0, 1), zs)
    u, s, v = torch.svd(usvt)
    q = torch.mm(v, torch.transpose(u, 0, 1))
    
    # alignment
    zsc = torch.norm(zc_bar) * torch.mm(zs_cap, q) + mean_c
    z = alpha*zc + (1-alpha)*zsc
    
    if(beta>0):
        
        # style image
        s1f = s1f.double()
        s1_channels, s1_width, s1_height = s1f.size(0), s1f.size(1), s1f.size(2)
        zs1 = s1f.view(s1_channels, -1)

        mean_s1 = torch.mean(zs1, 0)
        mean_s1 = mean_s1.unsqueeze(0).expand_as(zs1)
        zs1_bar = zs1 - mean_s1

        zs1_cap = zs1_bar/torch.norm(zs1_bar)

        # rotation
        us1vt = torch.mm(torch.transpose(zc, 0, 1), zs1)
        u1, s1, v1 = torch.svd(us1vt)
        q1 = torch.mm(v1, torch.transpose(u1, 0, 1))

        # alignment
        zs1c = torch.norm(zc_bar) * torch.mm(zs1_cap, q1) + mean_c
        z = alpha*zc + (1-alpha)*(beta*zs1c + (1-beta)*zsc)
    
    z = torch.reshape(z, (c_channels, c_width, c_height))
    return z.float().unsqueeze(0)
    
    
def moment_matching(cf, sf, alpha=0, beta=0, s1f=None):
    
    # content image
    cf = cf.double()
    c_channels, c_width, c_height = cf.size(0), cf.size(1), cf.size(2)
    zc = cf.view(c_channels, -1)
    
    mean_c = torch.mean(zc, 1)
    mean_c = mean_c.unsqueeze(1).expand_as(zc)
    var_c = torch.std(zc, 1).unsqueeze(1).expand_as(zc)
    
    # style image
    sf = sf.double()
    s_channels, s_width, s_height = sf.size(0), sf.size(1), sf.size(2)
    zs = sf.view(s_channels, -1)
    
    mean_s = torch.mean(zs, 1)
    mean_s = mean_s.unsqueeze(1).expand_as(zc)
    var_s = torch.std(zs, 1).unsqueeze(1).expand_as(zc)
    
    zsc = torch.div((zc - mean_c), var_c+1e-5) * var_s + mean_s
    z = alpha*zc + (1-alpha)*zsc
    
    if(beta>0):
        
        # style image
        s1f = s1f.double()
        s1_channels, s1_width, s1_height = s1f.size(0), s1f.size(1), s1f.size(2)
        zs1 = s1f.view(s1_channels, -1)

        mean_s1 = torch.mean(zs1, 1)
        mean_s1 = mean_s1.unsqueeze(1).expand_as(zc)
        var_s1 = torch.std(zs1, 1).unsqueeze(1).expand_as(zc)

        zs1c = torch.div((zc - mean_c), var_c+1e-5) * var_s1 + mean_s1
        z = alpha*zc + (1-alpha)*(beta*zs1c + (1-beta)*zsc)
    
    z = torch.reshape(z, (c_channels, c_width, c_height))
    return z.float().unsqueeze(0)
 

def stylize_ra(level, content, style, encoders, decoders, device, alpha=0, beta=0, style1=None):
    with torch.no_grad():
        if beta:
            cf = encoders[level](content).data.to(device=device).squeeze(0)
            sf = encoders[level](style).data.to(device=device).squeeze(0)
            s1f = encoders[level](style1).data.to(device=device).squeeze(0)
            csf = rigid_alignment(cf, sf, alpha, beta, s1f).to(device=device)
        else:
            cf = encoders[level](content).data.to(device=device).squeeze(0)
            sf = encoders[level](style).data.to(device=device).squeeze(0)
            csf = rigid_alignment(cf, sf, alpha).to(device=device)
        return decoders[level](csf)


def stylize_mm(level, content, style, encoders, decoders, device, alpha=0, beta=0, style1=None):
    with torch.no_grad():
        if beta:
            cf = encoders[level](content).data.to(device=device).squeeze(0)
            sf = encoders[level](style).data.to(device=device).squeeze(0)
            s1f = encoders[level](style1).data.to(device=device).squeeze(0)
            csf = moment_matching(cf, sf, alpha, beta, s1f).to(device=device)
        else:
            cf = encoders[level](content).data.to(device=device).squeeze(0)
            s0f = encoders[level](style).data.to(device=device).squeeze(0)
            csf = moment_matching(cf, s0f, alpha).to(device=device)
        return decoders[level](csf)