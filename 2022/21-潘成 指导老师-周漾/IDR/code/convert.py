from jittor.utils.pytorch_converter import convert
pytorch_code="""
torch.empty(n)

"""

jittor_code = convert(pytorch_code)
print(jittor_code)