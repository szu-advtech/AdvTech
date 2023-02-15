# In[]
import torch
import numpy as np
import cv2
from os.path import join
import os
import datetime

def unpack_svbrdf(inputs,less_channel=False):
    """
    将图片横向分为五个区域
    less_channel:normal以角度表示, roughness用单通道
    """
    # shape: B C H W
    # order: normals,  diffuse,  roughness,  specular
    out = []
    if less_channel:
        out.append(inputs[:,:2,:,:])
        out.append(inputs[:,2:5,:,:])
        out.append(inputs[:,5:6,:,:])
        out.append(inputs[:,6:,:,:])
    else:
        # print(normals.shape, diffuse.shape, roughness.shape, specular.shape)
        # return normals, diffuse, roughness, specular
        out = [inputs[:,i-3:i,:,:] for i in range(3,12+1,3)]
        out[0] = out[0]*2-1
    return out

# todo save batch
def save_sample(inputs,sample_dir,less_channel=False,input_img=None,save_name='sample.png'):
    ind = 0
    inputs = np.asarray(inputs.detach().cpu())
    inputs = unpack_svbrdf(inputs, less_channel=less_channel)
    if less_channel:
        inputs[0] = np.asarray(uv2normal(torch.tensor(inputs[0]-0.5)*torch.pi))
        inputs[2] = np.repeat(inputs[2],3,axis=1)
    inputs[0] = inputs[0]/2+0.5
    # print(type(inputs))
    if input_img is not None:
        inputs.insert(0,input_img)
    img = np.hstack([np.transpose(item[ind],[1,2,0]) for item in inputs])

    # img = np.hstack([np.transpose(inputs[ind,i-3:i,:,:],[1,2,0]) for i in range(3,12+1,3)])
    img = (img*255).astype('uint8')
    img = cv2.cvtColor(img ,cv2.COLOR_RGB2BGR)
    cv2.imwrite(join(sample_dir, save_name),img)
    return img

def set_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def log(logs, iter, writer):
    # if writer is None:
        # 默认train writer
        # writer = self.writer 
    
    # write to tensorboard
    d = dict(logs)
    for key in d:
        # self.writer.add_scalar('scalars/logs', d[key], global_step=d['iter']) 
        # self.writer.add_scalar('logs/'+key, d[key], global_step=iteration) 
        writer.add_scalar('logs/'+key, d[key], global_step=iter) 
    # self.log_iter += 1
    
    # with open(self.log_file, 'a') as f:
    #     f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

def tensor_show(tensor, show=True, **kargs):
    import matplotlib.pyplot as plt
    tmp = tensor.detach().cpu().permute(1,2,0)
    if show:
        plt.imshow(tmp, **kargs)
        plt.show()
    return tmp


def generate_normalized_random_direction(batchSize, lowEps = 0.001, highEps =0.05):
    # TODO really uniform?
    r1 = torch.rand([batchSize, 1]).uniform_(0.0 + lowEps, 1.0 - highEps)
    r2 =  torch.rand([batchSize, 1]).uniform_(0.0, 1.0)
    r = torch.sqrt(r1)
    phi = 2 * torch.pi * r2
       
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    # finalVec = torch.concat([x, y, z], axis=-1) #Dimension here is [batchSize, 3]
    finalVec = torch.cat([x, y, z], axis=-1) #Dimension here is [batchSize, 3]
    return finalVec


def generate_distance(batchSize):
    # TODO std or variance?
    gaussian = torch.randn([batchSize, 1]).normal_(0.5, 0.75) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (torch.exp(gaussian))


# def generate_random_scenes(batchSize, targets, outputs):    
def generate_diffuse_scenes(batchSize): 
    # output B 1 1 3 for B H W C   
    currentViewPos = generate_normalized_random_direction(batchSize)
    currentLightPos = generate_normalized_random_direction(batchSize)
    
    wi = currentLightPos
    # wi = tf.expand_dims(wi, axis=1)
    # wi = tf.expand_dims(wi, axis=1)
    wi = torch.reshape(wi, [batchSize, -1, 1, 1])
    
    wo = currentViewPos
    # wo = tf.expand_dims(wo, axis=1)
    # wo = tf.expand_dims(wo, axis=1)
    wo = torch.reshape(wo, [batchSize, -1, 1, 1])
    #[result, D_rendered, G_rendered, F_rendered, diffuse_rendered, specular_rendered]

    # return {'wi':wi, 'wo':wo}
    return [{'wi':torch.unsqueeze(wi[i],dim=0), 'wo':torch.unsqueeze(wo[i],dim=0)} for i in range(batchSize)]
    
    # return [{'wi':wi[i], 'wo':wo[i]} for i in range(batchSize)]
    # renderedDiffuse = tf_Render(targets,wi,wo)   
    
    # renderedDiffuseOutputs = tf_Render(outputs,wi,wo)#tf_Render_Optis(outputs,wi,wo)
    # return [renderedDiffuse, renderedDiffuseOutputs]


def generate_specular_scenes(batchSize,size=(256,256)):    
    # output B H W 3 for every pixel
    currentViewDir = generate_normalized_random_direction(batchSize)
    currentLightDir = currentViewDir * torch.reshape(torch.tensor([-1.0, -1.0, 1.0]), [1, -1])
    #Shift position to have highlight elsewhere than in the center.
    currentShift = torch.cat([torch.rand([batchSize, 2]).uniform_(-1.0, 1.0), torch.zeros([batchSize, 1]) + 0.0001], axis=-1)
    
    currentViewPos = torch.multiply(currentViewDir, generate_distance(batchSize)) + currentShift
    currentLightPos = torch.multiply(currentLightDir, generate_distance(batchSize)) + currentShift
    
    currentViewPos = torch.reshape(currentViewPos, [batchSize, 1, 1, -1])
    currentLightPos = torch.reshape(currentLightPos, [batchSize, 1, 1, -1])

    # print(currentShift.shape)
    # print(currentLightPos.shape)
    # print(currentLightDir.shape)
    # print(currentViewDir.shape)
    # print(currentViewPos.shape)
    surfaceArray = get_surface_array(size=size, concat_dim=-1)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    wi = wi.permute([0,3,1,2])
    wo = wo.permute([0,3,1,2])

    # return {'wi':wi, 'wo':wo}
    return [{'wi':torch.unsqueeze(wi[i],dim=0), 'wo':torch.unsqueeze(wo[i],dim=0)} for i in range(batchSize)]

    # return [{'wi':wi[i], 'wo':wo[i]} for i in range(batchSize)]
    # renderedSpecular = tf_Render(targets,wi,wo, includeDiffuse = a.includeDiffuse)           
    # renderedSpecularOutputs = tf_Render(outputs,wi,wo, includeDiffuse = a.includeDiffuse)#tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)
    # return [renderedSpecular, renderedSpecularOutputs]


# a=generate_normalized_random_direction(4, lowEps = 0.001, highEps =0.05)
# b=generate_distance(4)
# c=generate_random_scenes(4)
# print(a)
# print(b)
# print(c)

def get_surface_array(size=[256,256],concat_dim=-1):    
    x = torch.linspace(-1,1,size[0])
    y = torch.linspace(-1,1,size[1])
    # print(x.shape, y.shape)
    # import matplotlib.pyplot as plt
    # raise Exception()
    # plt.imshow(torch.meshgrid([x, y])[0])
    # plt.show()
    # plt.imshow(torch.meshgrid([x, y])[1])
    # plt.show()
    XX, YY= torch.meshgrid([x, y],indexing=None)
    # print(a)
    sur=torch.cat([XX.unsqueeze(concat_dim), YY.unsqueeze(concat_dim), torch.zeros(XX.shape).unsqueeze(concat_dim)],axis=concat_dim)
    # adapt to batch
    sur=sur.unsqueeze(0)
    # print(sur.shape)
    return sur

# a = get_surface_array(concat_dim=0) 
# out = generate_specular_scenes(4)
# print(out)
# print(out['wi'].shape)

def per_pix_dot(a,b,dim=0, keepdim=True):
    # C,H,W
    return torch.sum(a*b,dim=dim,keepdim=True)

# def normal2uv(normal):
#     """
#     convert nromal vector to angle coord
#     u:phi
#     v:theta
#     """
#     normal = torch.clip(normal,-1,1)
#     v = torch.acos(normal[:,:,-1])
#     u = torch.atan(normal[:,:,1]/(normal[:,:,0]+1e-11))
#     ind = normal[:,:,0]<0
#     # n = torch.acos(normal[:,:,-1])
#     # n = (normal[:,:,0]**2 + normal[:,:,1]**2)**0.5
#     # u = torch.acos(normal[:,:,0]/(n+1e-6))

#     # v[ind] = v[ind] + torch.pi/2
#     u[ind] = (torch.pi + u[ind]) % (2*torch.pi)
#     uv = torch.concat([u.unsqueeze(-1),v.unsqueeze(-1)],dim=-1)
#     return uv

def uv2normal(uv):
    uv = torch.clamp(uv,-torch.pi/2,torch.pi/2)
    # u = uv[:,:,0]
    # v = uv[:,:,-1]
    u = uv[:,0,:,:]
    v = uv[:,-1,:,:]
    z = torch.cos(v).unsqueeze(1)
    n = torch.sin(v)
    x = (torch.cos(u)*n).unsqueeze(1)
    y = (torch.sin(u)*n).unsqueeze(1)
    normal = torch.concat([x,y,z],dim=1)
    return torch.clamp(normal,-1,1)

def normal2uv(normal):
    """
    convert nromal vector to angle coord
    """
    normal = torch.clip(normal,-1,1)
    x = normal[:,0,:,:]
    y = normal[:,1,:,:]
    z = normal[:,2,:,:]
    # x = normal[:,:,0]
    # y = normal[:,:,1]
    # z = normal[:,:,2]
    ind = x<0
    v = torch.acos(z)
    n = (x**2+y**2)**0.5
    # u = torch.acos(x/torch.asin(z))
    # u = torch.acos(x/n)
    u = torch.asin(y/(n+1e-12))
    v[ind] = -v[ind]
    u[ind] = -u[ind]
    uv = torch.concat([u.unsqueeze(1),v.unsqueeze(1)],dim=1)
    return angle_clamp(uv)

def angle_clamp(uv):
    """
    TODO 超出范围的通过三角公式转到区间内?
    """
    return torch.clamp(uv,-torch.pi/2,torch.pi/2)
# def normal2uv(normal):
#     """
#     convert nromal vector to angle coord
#     """
#     normal = torch.clip(normal,-1,1)
#     v = torch.asin(normal[:,:,-1])
#     ind = normal[:,:,1]<0
#     # n = torch.acos(normal[:,:,-1])
#     n = (normal[:,:,0]**2 + normal[:,:,1]**2)**0.5
#     u = torch.acos(normal[:,:,0]/(n+1e-6))

#     v[ind] = v[ind] + torch.pi/2
#     u[ind] = torch.pi - u[ind]
#     uv = torch.concat([u.unsqueeze(-1),v.unsqueeze(-1)],dim=-1)
#     return uv

# def uv2normal(uv):
#     uv = torch.clip(uv,0,torch.pi)
#     u = uv[:,:,0]
#     v = uv[:,:,-1]
#     z = torch.sin(v).unsqueeze(-1)
#     n = torch.cos(v)
#     x = (torch.cos(u)*n).unsqueeze(-1)
#     y = (torch.sin(u)*n).unsqueeze(-1)
#     normal = torch.concat([x,y,z],dim=-1)
#     return normal

"""
# In[]
im=cv2.imread(r'E:\CGM\PS\photometric-stereo-outlier-rejection-main\buddhaPNG_Normal.png')
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)/255.0*2-1
uv = normal2uv(torch.tensor(im))
# torch.zeros([512,612,1]).shape
uv_show = torch.concat([uv/torch.pi,torch.zeros([512,612,1])],dim=-1)
# print(uv.shape)
normal = uv2normal(uv)

# %%
import matplotlib.pyplot as plt
plt.imshow(uv_show)
plt.show()
plt.imshow(normal/2+0.5)
plt.show()
plt.imshow(im/2+0.5)
plt.show()
# %%
mask = im.sum(-1)>0
plt.hist(uv[:,:,0][mask].flatten(),bins=100)
plt.show()
plt.hist(uv[:,:,1][mask].flatten(),bins=100)
plt.show()
# %%
"""
