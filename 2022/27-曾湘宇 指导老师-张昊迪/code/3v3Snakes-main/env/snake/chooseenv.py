# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/9/11 11:17 上午
# 描述：选择运行环境，需要维护env/__ini__.py && config.json（存储环境默认参数）

import json

import os

import snakes


def make(env_type, conf=None):
    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not conf:
        with open(file_path) as f:
            '''
            json.loads()：解析一个有效的JSON字符串并将其转换为Python字典
            json.load()：从一个文件读取JSON类型的数据，然后转转换成Python字典
            '''
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']
    '''
    getattr() 函数用于返回一个对象属性值。
    getattr(object, name[, default])
        >>>class A(object):
        ...     bar = 1
        ... 
        >>> a = A()
        >>> getattr(a, 'bar')        # 获取属性 bar 值
        1
        >>> getattr(a, 'bar2')       # 属性 bar2 不存在，触发异常
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        AttributeError: 'A' object has no attribute 'bar2'
        >>> getattr(a, 'bar2', 3)    # 属性 bar2 不存在，但设置了默认值
        3
        >>>
    '''
    return getattr(snakes, class_literal)(conf) #(conf)什么意思


if __name__ == "__main__":
    res = make("snakes_3v3")
    print(res)
