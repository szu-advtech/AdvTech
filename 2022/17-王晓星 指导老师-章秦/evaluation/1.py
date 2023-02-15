



# f=open('test_human_0.txt', "r", encoding='utf-8')
# f1=open('test_human_2.txt', "w", encoding='utf-8')
# for line in f:
#     # post = line.find('标题：')
#     # if post != -1:
#     #     continue
#     line=line.replace('。','\n').replace(':','\n').replace('；','\n').replace('？','\n').replace('!','\n').replace('.','\n')
#     if len(line) <=4:
#         continue
#     f1.write(line)
# f.close()
# f1.close()
# sens=[]
# f = open('test_human.txt', "r", encoding='utf-8')
# for line in f:
#     post = line.find('标题：')
#     if post != -1:
#         continue
#     sens.append(line.replace('\n', '').replace('“', '').replace('”', ''))
# print(sens)
with open('eval.txt','r',encoding='utf-8') as f_obj:
    readthings=f_obj.read()
readthings=readthings.split('标题：')
a=0
count=a+len(readthings)
print(count-1)
num=1
for i in range(1,count+1):
    with open('../data/eval_data/%d.txt'%i,'w+',encoding='utf-8') as f_obj:
        f_obj.write(readthings[num])
        num+=1