from IPy import IP


def tostr(tr):
    str_address=[]
    for ip in tr:
        str_address.append(''.join(l + ':' * (n % 4 == 3) for n, l in enumerate([f'{i:x}' for i  in ip ]))[:-1])
    for i,ip in enumerate(str_address):
        str_address[i]=IP(ip).strCompressed()
    print(str_address[0])
    print(type(str_address[0]))
    with open('./test.txt','w') as f:
        for ip in str_address:
            f.write(ip+'\n')