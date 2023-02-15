def Electricity(args):
    args.is_training = 1
    args.root_path = './dataset/electricity/'  # 数据集文件夹路径
    args.data_path = 'electricity.csv'  # 数据集文件
    args.model_id = 'ECL_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'custom'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 321  # encoder输入大小
    args.dec_in = 321  # decoder输入大小
    args.c_out = 321  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def ETTh1(args):
    args.is_training = 1
    args.root_path = './dataset/ETT-small/'  # 数据集文件夹路径
    args.data_path = 'ETTh1.csv'  # 数据集文件
    args.model_id = 'ETTh1_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'ETTh1'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 7  # encoder输入大小
    args.dec_in = 7  # decoder输入大小
    args.c_out = 7  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def ETTh2(args):
    args.is_training = 1
    args.root_path = './dataset/ETT-small/'  # 数据集文件夹路径
    args.data_path = 'ETTh2.csv'  # 数据集文件
    args.model_id = 'ETTh2_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'ETTh2'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 7  # encoder输入大小
    args.dec_in = 7  # decoder输入大小
    args.c_out = 7  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def ETTm1(args):
    args.is_training = 1
    args.root_path = './dataset/ETT-small/'  # 数据集文件夹路径
    args.data_path = 'ETTm1.csv'  # 数据集文件
    args.model_id = 'ETTm1_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'ETTm1'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 7  # encoder输入大小
    args.dec_in = 7  # decoder输入大小
    args.c_out = 7  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def ETTm2(args):
    args.is_training = 1
    args.root_path = './dataset/ETT-small/'  # 数据集文件夹路径
    args.data_path = 'ETTm2.csv'  # 数据集文件
    args.model_id = 'ETTm2_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'ETTm2'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 7  # encoder输入大小
    args.dec_in = 7  # decoder输入大小
    args.c_out = 7  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def Exchange(args):
    args.is_training = 1
    args.root_path = './dataset/exchange_rate/'  # 数据集文件夹路径
    args.data_path = 'exchange_rate.csv'  # 数据集文件
    args.model_id = 'Exchange_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'custom'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 8  # encoder输入大小
    args.dec_in = 8  # decoder输入大小
    args.c_out = 8  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def Illness(args):
    args.is_training = 1
    args.root_path = './dataset/illness/'  # 数据集文件夹路径
    args.data_path = 'national_illness.csv'  # 数据集文件
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'custom'  # 数据类型
    args.features = 'M'  # 预测类别
    args.seq_len = 36
    args.label_len = 18
    args.model_id = 'illness_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 7  # encoder输入大小
    args.dec_in = 7  # decoder输入大小
    args.c_out = 7  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    return

def Traffic(args):
    args.is_training = 1
    args.root_path = './dataset/traffic/'  # 数据集文件夹路径
    args.data_path = 'traffic.csv'  # 数据集文件
    args.model_id = 'traffic_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'custom'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 862  # encoder输入大小
    args.dec_in = 862  # decoder输入大小
    args.c_out = 862  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    args.train_epochs = 2
    return

def Weather(args):
    args.is_training = 1
    args.root_path = './dataset/weather/'  # 数据集文件夹路径
    args.data_path = 'weather.csv'  # 数据集文件
    args.model_id = 'weather_' + str(args.seq_len) + '_' + str(args.pred_len)  # 模型id
    args.model = 'Autoformer'  # 选择使用模型
    args.data = 'custom'  # 数据类型
    args.features = 'M'  # 预测类别
    args.e_layers = 2  # encoder层数
    args.d_layers = 1  # decoder层数
    args.factor = 3  #
    args.enc_in = 21  # encoder输入大小
    args.dec_in = 21  # decoder输入大小
    args.c_out = 21  # 输出长度
    args.des = 'Exp'  #
    args.itr = 1  # 实验次数
    args.train_epochs = 2
    return