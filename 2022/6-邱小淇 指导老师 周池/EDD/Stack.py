"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

class Stack:
    def __init__(self):
        self.items = []
    """
    栈中的函数：
    isEmpty()判断栈是否为空
    push()把元素入栈
    pop()把最后一个元素删除并返回
    peek()显示最后一个元素（栈顶）
    size()返回栈的大小
    """
    def isEmpty(self):
        return self.items == []

    def put(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def __str__(self):
        res = 'from bottom to top : '
        res += str([str('-').join(str(k)) for k in self.items])
        return res