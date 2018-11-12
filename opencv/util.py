#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from os import path 

d = path.dirname(__file__)  #返回当前文件所在的目录
parent_path = os.path.dirname(d)
# abspath = path.abspath(d) #返回d所在目录规范的绝对路径
image_path = parent_path + "/images/"

def getImagePath(name):
    return parent_path + "/images/" + name

