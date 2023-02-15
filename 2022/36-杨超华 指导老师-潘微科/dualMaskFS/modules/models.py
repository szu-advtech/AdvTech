import torch
from modules.layers import MultiLayerPerceptron, FactorizationMachine, FeaturesLinear, FeatureEmbedding
import modules.layers as layer


class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        print(self.field_num)
        print(self.feature_num)
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def reg(self):
        return 0.0
    
class FM(BasicModel):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        #self.linear = FeaturesLinear(self.feature_num)
        self.fm = FactorizationMachine(reduce_sum=True)
    
    def forward(self, x):
        x_embedding = self.embedding(x)
        #output_linear = self.linear(x)
        output_fm = self.fm(x_embedding)
        #print(output_linear.shape)
        #print(output_fm.shape)
        #logit = output_linear +  output_fm
        logit = output_fm
        return logit

class DNN(BasicModel):
    def __init__(self, opt):
        super(DNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num*self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)


    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn
        return logit

class DeepFM(FM):
    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num*self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)


    def forward(self, x):
        x_embedding = self.embedding(x)
        #output_linear = self.linear(x)

        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        #logit = output_dnn +  output_fm + output_linear
        logit = output_dnn +  output_fm
        return logit

class DeepCrossNet(BasicModel):
    def __init__(self, opt):
        super(DeepCrossNet, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross,output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit


class InnerProductNet(BasicModel):
    def __init__(self, opt):
        super(InnerProductNet, self).__init__(opt)
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim + \
                       int(self.field_num * (self.field_num -1)/2)
        self.inner = layer.InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num*self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn= torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit


