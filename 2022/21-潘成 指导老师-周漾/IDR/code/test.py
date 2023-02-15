import jittor as jt

# A = jt.array((1., 2.0)).reshape(1, 2)
# B = jt.array((1.0, 2.0, 3., 4.)).reshape(2, 2)
# print("A：   ",A)
# print("B：   ",B)

# C = A @ B
# print("C：   ",C)

# sum = C.sum()
# print("sum = ", sum)
# dA, dB = jt.grad(C, [A, B])
# print("dA, dB: ", dA, dB)
# A.stop_grad()
# dA, dB = jt.grad(sum * sum, [A, B])
# print("dA, dB: ", dA, dB)

# clamp
# print(jt.arange(100, 0, -1))

# def arange(start=0, end=None, step=1,dtype=None):
#     print("start, end, step, ", start, end, step)
#     if end is None:
#         end,start = start,0
#     l = round((end-start)//step)+1
#     # print("l = :            ", l)
#     if (l-1)*step+start>=end:
#         l-=1
        
#     # print("lll = :            ", l)
#     x = jt.index((l,),0)
#     x = x*step+start
#     if dtype is not None:
#         x= x.cast(dtype)
#     return x
# steps = jt.zeros(100).uniform_(0.0, 1.0)

# A = steps.unsqueeze(0).repeat(2004,1)
# steps = jt.zeros(10).uniform_(0.0, 1.0)
# print("steps.unsqueeze(0).shape = ", steps.unsqueeze(0).shape)
# steps = steps.unsqueeze(0).repeat(14, 1)

A = jt.randn(2, 3)
# print(jt.arg_reduce(A, 'min', dim = 1, keepdim=False))
# print(jt.arg_reduce(A, 'min', dim=1, keepdims=False))
B = A.sum()
print(A)
print(B)
print(jt.grad(B, A))