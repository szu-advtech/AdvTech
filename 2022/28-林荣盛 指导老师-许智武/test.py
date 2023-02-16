import torch

if __name__=='__main__':
    a=torch.rand(1,2,4,2)
    print(a)
    b=torch.sum(a,dim=3,keepdim=True)
    print(b)
    b=torch.sum(b,dim=2,keepdim=True)
    print(b)

    weights = a.div(b)  # (b,l,4,g)
    weights = weights * 4
    print('*' * 30)
    print(weights)
    print('*' * 30)
    weights = weights.unsqueeze(-1)
    print('*'*30)
    print(weights)
    print('*' * 30)
    gaz_embeds=torch.rand(1,2,4,2,5)
    print('*' * 30)
    print(gaz_embeds)
    print('*' * 30)
    gaz_embeds = weights * gaz_embeds  # (b,l,4,g,e)
    print('*' * 30)
    print(gaz_embeds)
    print('*' * 30)
    gaz_embeds = torch.sum(gaz_embeds, dim=3)  # (b,l,4,e)
    print('*' * 30)
    print(gaz_embeds)

    text='屠呦呦，女，1930年12月30日出生于浙江宁波 [78]  ，汉族，' \
         '中共党员，药学家。1951年考入北京大学医学院药学系生药专业。 [1-2]  ' \
         '1955年毕业于北京医学院（今北京大学医学部）。毕业后接受中医培训两年半，' \
         '并一直在中国中医研究院（2005年更名为中国中医科学院）工作，期间晋升为硕士生导师、' \
         '博士生导师。现为中国中医科学院首席科学家， [3-5]   终身研究员兼首席研究员 [6]  ，' \
         '青蒿素研究开发中心主任，博士生导师，共和国勋章获得者。 [7] '
    text=list(text)
    print(len(text))
    text=[s for s in text if s.strip() != '' ]
    print(len(text))
    with open('./data/test1.char','w',encoding='utf8') as fp:
        for s in text:
            fp.write(s+'\n')
        fp.write(' ')

    with open('./data/test1.char', 'r', encoding='utf8') as fp:
        text=fp.readlines().strip()
    print(text)