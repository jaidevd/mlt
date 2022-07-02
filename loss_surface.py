# coding: utf-8
import numpy as np
x = np.random.rand(7,)
x
x = np.random.randint(0,10, size=(7,))
x
m = np.arange()
m = np.arange(5)
c = np.arange(-5, 0)
c
m
mx = x.reshape(-1, 1) @ m.reshape(1, -1)
mx
mx.shape
c
import pandas as pd
mx = pd.DataFrame(mx, index=[f'x{i}' for i in range(len(x))], columns=[f'm_{i}' for i in range(len(m))])
mx
c
Z = np.zeros((7, 5, 5))
get_ipython().run_line_magic('pinfo', 'np.tile')
xmin = mx.values[:2, :2]
xmin
get_ipython().run_line_magic('pinfo', 'np.tile')
np.tile(xmin, (2, 2, 3))
XM = np.tile(xmin, (2, 2, 3))
XM.shape
XM = np.tile(xmin, (1, 2, 3))
XM.shape
xmin
get_ipython().run_line_magic('pinfo', 'np.tile')
get_ipython().run_line_magic('pinfo', 'np.repeat')
np.repeat(xmin, [1, 2, 3], np.newaxis)
mxin
xmin
get_ipython().run_line_magic('pinfo', 'np.stack')
np.stack([xmin] * 3, axis=np.newaxis)
xmin
np.tile(xmin, (1, 1, 3))
_.shape
np.tile(xmin, (1, 1, 2))
_.shape
np.tile(xmin, (2, 1, 2))
_.shape
np.tile(xmin, (2, 1, 3))
_.shape
np.tile(xmin, (3, 1, 1))
mx
MX
MX = mx.values
MX
c.shape
MX_CShape = np.tile(MX, c.shape + (1, 1))
MX_CShape
MX_CShape + c
c
MX
[MX + _c for _c in c]
MX_CShape
MX_CShape
MX_CShape.shape
c.shape
np.tile(c, (5, 7, 5))
np.tile(c, (1, 7, 5))
np.tile(c, (5, 76))
np.tile(c, (5, 7))
np.tile(c, (5, 7, 1))
MX_CShape
Cnew = np.tile(c, (5, 7, 1))
MX_CShape + Cnew
[MX + _c for _c in c]
c
CNew
Cnew
np.ones((5, 7, 5)) * c
np.ones((5, 7, 5)) * c.reshape((5, 1, 1))
Cnew = np.ones((5, 7, 5)) * c.reshape((5, 1, 1))
MX_CShape + Cnew
[MX + _c for _c in c]
Z = MX_CShape + Cnew
X
Z
m
x
M
C
c
M, C = 2, -3
