import torch
import torch.nn as nn

class Novel_Residual_Block(nn.Module):
    def __init__(self, dim):
        super(Novel_Residual_Block, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=0),
                       nn.BatchNorm2d(num_features=dim),
                       nn.LeakyReLU(0.2),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=(1,1), padding=0, groups=dim),
                       nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 1), stride=(1,1), padding=0, groups=1),
                       nn.BatchNorm2d(num_features=dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Coarse_Generator(nn.Module):
    def __init__(self, input_nc, ncf=64, n_downsampling=2, n_blocks=9, norm_layer=nn.BatchNorm2d):
        super(Coarse_Generator, self).__init__()
        activation = nn.LeakyReLU(0.2)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels=input_nc, out_channels=ncf, kernel_size=(7,7), stride=(1,1),padding=0),
                 norm_layer(ncf),
                 activation]
        # downsample
        for i in range(n_downsampling):
            down_filters = ncf * pow(2, i) * 2
            model += [nn.Conv2d(in_channels=ncf*(i+1), out_channels=down_filters, kernel_size=(3,3), stride=(2,2), padding=1),
                      norm_layer(num_features=down_filters), activation]

        # novel resnet blocks
        res_filters = pow(2, n_downsampling)
        dim = res_filters * ncf
        for i in range(n_blocks):
            model += [Novel_Residual_Block(dim)]

        # upsample
        for i in range(n_downsampling):
            up_filters = int(ncf * pow(2, (n_downsampling - i)) / 2)
            model += [nn.ConvTranspose2d(up_filters * 2, up_filters, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
                      norm_layer(num_features=up_filters), activation]

        self.model = nn.Sequential(*model)

        self.reflection_pad_2d = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=0)
        self.tanh = nn.Tanh()
        #print(type(model))

    def forward(self, input):
        feature_out = self.model(input)
        output = self.reflection_pad_2d(feature_out)
        output = self.conv(output)
        output = self.tanh(output)
        return output,feature_out

# coarse_generator = Coarse_Generator(input_nc=3)
# print(coarse_generator)
#
# ############################coarse generator##############################
# coarse_input = torch.arange(256*256*3, dtype=torch.float).reshape(1, 3, 256, 256)
# coarse_output,coarse_feature_out = coarse_generator(coarse_input)
# print(coarse_output.shape)
# print(type(coarse_output))
# print(coarse_feature_out.shape)
# print(type(coarse_feature_out))

# coarse generator解析过程
# print(input.shape)
#
# m = nn.ReflectionPad2d(1) #填充顺序是左-右-上-下；1代表填充1层
# print(m(input).shape)
#
# ncf = 64
# mm = nn.Conv2d(in_channels=3, out_channels=ncf, kernel_size=(7,7), stride=(1,1))
# print(mm(m(input)).shape)
#
# mmm = nn.BatchNorm2d(num_features=ncf)
# print(mmm(mm(m(input))).shape)
#
# mmmm = nn.LeakyReLU(0.2)
# print(mmmm(mmm(mm(m(input)))).shape)
#
# re = mmmm(mmm(mm(m(input))))
#
# n_downsampling = 2
# for i in range(n_downsampling):
#     down_filters = ncf * pow(2, i) * 2
#     print(down_filters)
#     n = nn.Conv2d(in_channels=ncf*(i+1), out_channels=down_filters, kernel_size=(3,3), stride=(2,2), padding=1)
#     print(n(re).shape)
#     n_n = nn.BatchNorm2d(num_features=down_filters)
#     print(n_n(n(re)).shape)
#     n_n_n = nn.LeakyReLU(0.2)
#     print(n_n_n(n_n(n(re))).shape)
#     re = n_n_n(n_n(n(re)))
#
# print("-----------------------------------------------")
# re_new1 =re
#
# def novel_residual_block(re_new1, all):
#     re = re_new1
#     print(re.shape)
#
#     m = nn.ReflectionPad2d(1)  # 填充顺序是左-右-上-下；1代表填充1层
#     re = m(re)
#     print(re.shape)
#
#     mm = nn.Conv2d(in_channels=all, out_channels=all, kernel_size=(3, 3), stride=(1, 1), padding=0)
#     re = mm(re)
#     print(re.shape)
#
#     mmm = nn.BatchNorm2d(num_features=all)
#     re = mmm(re)
#     print(re.shape)
#
#     mmmm = nn.LeakyReLU(0.2)
#     re = mmmm(re)
#     print(re.shape)
#
#     mmmmm = nn.ReflectionPad2d(1)  # 填充顺序是左-右-上-下；1代表填充1层
#     re = mmmmm(re)
#     print(re.shape)
#
#     mmmmmm = nn.Conv2d(in_channels=all, out_channels=all, kernel_size=(3, 3), stride=1, padding=0, groups=all)
#     re = mmmmmm(re)
#     print(re.shape)
#     print("-----------------------------------------------")
#     mmmmmmm = nn.Conv2d(in_channels=all, out_channels=all, kernel_size=(1, 1), stride=1, padding=0, groups=1)
#     re = mmmmmmm(re)
#     print(re.shape)
#
#     mmmmmmmm = nn.BatchNorm2d(num_features=all)
#     re = mmmmmmmm(re)
#     print(re.shape)
#
#     re = re_new1 + re
#     print(re.shape)
#     print("-----------------------------------------------")
#     return re
#
# ### novel resnet blocks
# n_blocks = 9
# res_filters = pow(2,n_downsampling)
# all = res_filters*ncf
# for i in range(n_blocks):
#     re_new1 = novel_residual_block(re_new1,all)
# re_new2 = re_new1
# print(re_new2.shape)
#
# print("-------------------------upupupupupupupupupup---------------------")
# for i in range(n_downsampling):
#     up_filters = int(ncf * pow(2, (n_downsampling - i)) / 2)
#     # X = Conv2DTranspose(filters=up_filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
#     #                         kernel_initializer=RandomNormal(stddev=0.02))(X)
#     z = nn.ConvTranspose2d(up_filters * 2, up_filters, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
#     print(z(re_new2).shape)
#     z_z = nn.BatchNorm2d(num_features=up_filters)
#     print(z_z(z(re_new2)).shape)
#     z_z_z = nn.LeakyReLU(0.2)
#     print(z_z_z(z_z(z(re_new2))).shape)
#     re_new2 = z_z_z(z_z(z(re_new2)))
#
# feature_out = re_new2
# print(re_new2.shape)
# print("feature_out",feature_out.shape)
#
# print("-------------------------last---------------------")
# l = nn.ReflectionPad2d(3)
# print(l(re_new2).shape)
#
# ll = nn.Conv2d(in_channels=ncf, out_channels=1, kernel_size=(7,7), stride=(1,1), padding=0)
# print(ll(l(re_new2)).shape)
#
# lll = nn.Tanh()
# print(lll(ll(l(re_new2))).shape)

class Fine_Generator(nn.Module):
    def __init__(self, input_nc, nff=32, n_coarse_gen=1, n_blocks=3, norm_layer=nn.BatchNorm2d):
        super(Fine_Generator, self).__init__()
        activation = nn.LeakyReLU(0.2)
        model1 = []
        # downsample
        i = 1
        down_filters = nff * (2 ** (n_coarse_gen - i))
        model1 += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=input_nc, out_channels=down_filters, kernel_size=(7, 7), stride=(1, 1), padding=0),
                  norm_layer(num_features=down_filters),
                  activation,
                  nn.Conv2d(in_channels=down_filters, out_channels=down_filters*2, kernel_size=(3, 3), stride=(2, 2), padding=1),
                  norm_layer(num_features=down_filters*2),
                  activation]
        self.model1 = nn.Sequential(*model1)
        # print(type(model1))

        # Connection from Coarse Generator

        model2 = []
        # novel resnet blocks
        res_filters = nff * (2 ** (n_coarse_gen - i)) * 2
        for j in range(n_blocks):
            model2 += [Novel_Residual_Block(res_filters)]

        # upsample
        up_filters = nff * (2**(n_coarse_gen-i))
        model2 += [nn.ConvTranspose2d(up_filters * 2, up_filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                   norm_layer(num_features=up_filters),
                   activation]
        self.model2 = nn.Sequential(*model2)

        self.reflection_pad_2d = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=0)
        self.tanh = nn.Tanh()

        #print(type(model2))

    def forward(self, input):
        input_input, feature_out = input
        input_add = feature_out + self.model1(input_input)
        output = self.model2(input_add)
        output = self.reflection_pad_2d(output)
        output = self.conv(output)
        output = self.tanh(output)
        return output

# fine_generator = Fine_Generator(input_nc=3)
# print(fine_generator)
#
# ############################fine generator##############################
# fine_input = torch.arange(512*512*3, dtype=torch.float).reshape(1, 3, 512, 512)
# fine_output = fine_generator(fine_input,coarse_feature_out)
# print(fine_output.shape)
# print(type(fine_output))

class Discriminator(nn.Module):
    def __init__(self, input_cf_nc, input_ffa_nc, n_downsampling, hw, ndf=32, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        activation = nn.LeakyReLU(0.2)
        model = []
        for i in range(n_downsampling): #n_downsampling=0 不进行循环 n_downsampling=1 循环1次
            model += [nn.AvgPool2d(kernel_size=(3,3),stride=(2,2),padding=1)]

        model += [nn.Conv2d(in_channels=input_cf_nc+input_ffa_nc, out_channels=ndf, kernel_size=(4, 4), stride=(2, 2), padding=1),
                  activation]

        down_filters = min(ndf * 2, 512)
        for i in range(1, n_layers):
            #print(down_filters)
            model += [nn.Conv2d(in_channels=min(ndf * 2, ndf * i), out_channels=down_filters, kernel_size=(4,4), stride=(2,2), padding=1),
                      norm_layer(num_features=down_filters),
                      activation]

        model += [nn.Conv2d(in_channels=down_filters, out_channels=down_filters, kernel_size=(3, 3), stride=(1, 1), padding=1),
                  norm_layer(num_features=down_filters),
                  activation,
                  nn.Conv2d(in_channels=down_filters, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)]

        self.model = nn.Sequential(*model)
        #print(type(model))
        self.linear = nn.Linear(in_features=hw,out_features=hw)

    def forward(self, input):
        input_cf, input_ffa = input
        input = torch.cat((input_cf, input_ffa), 1)
        #print(input.shape)
        output = self.model(input)
        #print(output.shape)

        batchsize = output.shape[0]
        C = output.shape[1]
        H = output.shape[2]
        W = output.shape[3]
        #print(output.shape)
        output = self.linear(output)
        #print(output.shape)
        output = output.view(batchsize, C, H, W)
        #print(output.shape)
        return output

# a = torch.arange(512*512*3, dtype=torch.float).reshape(1, 3, 512, 512)
# b = torch.arange(512*512*3, dtype=torch.float).reshape(1, 3, 512, 512)
# c = torch.cat((a,b),1)
# print(c.shape)

# discriminator1 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=0)
# discriminator2 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=1)
# # discriminator3 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=0)
# # discriminator4 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=1)
# print(discriminator1)
# from torchsummary import summary
# summary(discriminator1,input_size=[(3,512,512),(1,512,512)])  # 输出网络结构
#
# ############################discriminator generator##############################
# fine_label = torch.arange(512*512*1, dtype=torch.float).reshape(1, 1, 512, 512)
# coarse_label = torch.arange(256*256*1, dtype=torch.float).reshape(1, 1, 256, 256)
# # D1 Fine
# d1_output = discriminator1(fine_input,fine_label)
# print(d1_output.shape)
# print(type(d1_output))
# # D2 Fine
# d2_output = discriminator2(fine_input,fine_label)
# print(d2_output.shape)
# print(type(d2_output))
# # D1 Coarse
# d3_output = discriminator3(coarse_input,coarse_label)
# print(d3_output.shape)
# print(type(d3_output))
# # D2 Coarse
# d4_output = discriminator4(coarse_input,coarse_label)
# print(d4_output.shape)
# print(type(d4_output))

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

# class Fundus2Angio_Original(nn.Module):
#     def __init__(self, input_g_nc=3, input_cf_nc=3, input_ffa_nc=1):
#         super(Fundus2Angio_Original, self).__init__()
#
#         coarse_generator = [Coarse_Generator(input_nc=input_g_nc)]
#         self.coarse_generator = nn.Sequential(*coarse_generator)
#
#         fine_generator = [Fine_Generator(input_nc=input_g_nc)]
#         self.fine_generator = nn.Sequential(*fine_generator)
#
#         d1 = [Discriminator(input_cf_nc=input_cf_nc, input_ffa_nc=input_ffa_nc, n_downsampling=0)]
#         self.d1 = nn.Sequential(*d1)
#         freeze(self.d1)
#         d2 = [Discriminator(input_cf_nc=input_cf_nc, input_ffa_nc=input_ffa_nc, n_downsampling=1)]
#         self.d2 = nn.Sequential(*d2)
#         freeze(self.d2)
#         d3 = [Discriminator(input_cf_nc=input_cf_nc, input_ffa_nc=input_ffa_nc, n_downsampling=0)]
#         self.d3 = nn.Sequential(*d3)
#         freeze(self.d3)
#         d4 = [Discriminator(input_cf_nc=input_cf_nc, input_ffa_nc=input_ffa_nc, n_downsampling=1)]
#         self.d4 = nn.Sequential(*d4)
#         freeze(self.d4)
#
#     def forward(self, input_cf, input_ffa, input_cf_half, input_ffa_half):
#         coarse_output, coarse_feature_out = self.coarse_generator(input_cf_half)
#         fine_output = self.fine_generator(input_cf, coarse_feature_out)
#
#         # D1 Fine
#         d1_output = self.d1(input_cf, fine_output)
#         # print(d1_output.shape)
#         # print(type(d1_output))
#
#         # D2 Fine
#         d2_output = self.d2(input_cf, fine_output)
#         # print(d2_output.shape)
#         # print(type(d2_output))
#
#         # D1 Coarse
#         d3_output = self.d3(input_cf_half, coarse_output)
#         # print(d3_output.shape)
#         # print(type(d3_output))
#
#         # D2 Coarse
#         d4_output = self.d4(input_cf_half, coarse_output)
#         # print(d4_output.shape)
#         # print(type(d4_output))
#
#         return coarse_output,fine_output,d1_output,d2_output,d3_output,d4_output

class Fundus2Angio(nn.Module):
    def __init__(self, coarse_generator,fine_generator, d1, d2, d3, d4):
        super(Fundus2Angio, self).__init__()

        self.coarse_generator = nn.Sequential(coarse_generator)

        self.fine_generator = nn.Sequential(fine_generator)

        self.d1 = nn.Sequential(d1)
        freeze(self.d1)

        self.d2 = nn.Sequential(d2)
        freeze(self.d2)

        self.d3 = nn.Sequential(d3)
        freeze(self.d3)

        self.d4 = nn.Sequential(d4)
        freeze(self.d4)

    def forward(self, input):
        input_cf, input_ffa, input_cf_half, input_ffa_half, coarse_feature_out = input
        coarse_output, _ = self.coarse_generator(input_cf_half)
        fine_output = self.fine_generator((input_cf, coarse_feature_out))

        # D1 Fine
        d1_output = self.d1((input_cf, fine_output))
        # print(d1_output.shape)
        # print(type(d1_output))

        # D2 Fine
        d2_output = self.d2((input_cf, fine_output))
        # print(d2_output.shape)
        # print(type(d2_output))

        # D1 Coarse
        d3_output = self.d3((input_cf_half, coarse_output))
        # print(d3_output.shape)
        # print(type(d3_output))

        # D2 Coarse
        d4_output = self.d4((input_cf_half, coarse_output))
        # print(d4_output.shape)
        # print(type(d4_output))

        return coarse_output,fine_output,d1_output,d2_output,d3_output,d4_output

def Fundus2Angio_Gan(coarse_generator,fine_generator, d1, d2, d3, d4):
    # Discriminator NOT trainable
    freeze(d1)
    freeze(d2)
    freeze(d3)
    freeze(d4)

    gan_model = Fundus2Angio(coarse_generator,fine_generator, d1, d2, d3, d4)
    #coarse_output, fine_output, d1_output, d2_output, d3_output, d4_output = gan_model(input_cf, input_ffa, input_cf_half, input_ffa_half, coarse_feature_out)
    return gan_model
