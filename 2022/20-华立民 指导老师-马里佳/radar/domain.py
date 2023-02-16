import radar.environments.predator_prey as predator_prey

def make(domain, params={}):
    if domain.startswith("PredatorPrey-"):   #验证输入参数以PredatorPrey-开头，则调用函数
        return predator_prey.make(domain, params) #根据输入参数，创建对应的PP环境
    raise ValueError("Environment '{}' unknown or not published yet".format(domain))  #否则抛出一个错误