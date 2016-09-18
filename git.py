# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
class kernelfunction():
    def __init__(self,theta1,theta2,theta3,theta4):
        """
            kernel = Θ1*exp(-Θ2/2*||x1-x2||^2) + Θ3 + Θ4*<x1|x2>
            defaultはガウシアンカーネルを使用
            別途追加すれば別のカーネルに変更することもできるので
            """
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4
        self.norm = "gaussian" # default
    def _kernel(self,x1,x2):
        if self.norm == "gaussian":
            return np.exp(-self.theta2/2.*np.inner(x1-x2,x1-x2))
        if self.norm == "ornstein process": #オルンシュタインウーレンベック過程
            return np.exp(-self.theta2*np.power(np.inner(x1-x2,x1-x2),.5))
    def kernel(self,x1,x2):
        """
            calculate kernel
            x1,x2: numpy.array, has same dimention
            """
        val = self.theta1 * self._kernel(x1,x2) + self.theta3 + self.theta4 * (np.inner(x1,x2))
        return val

#kernel1 = kernelfunction(1,4.,0.,0.)#thata1=1,theta2=4 よくあるガウスカーネルになってます。
#kernel1 = kernelfunction(1,64.,0.,0.)#thata1=1,theta2=4 よくあるガウスカーネルになってます。
kernel1 = kernelfunction(1,4.,10.,0.)#thata1=1,theta2=4 よくあるガウスカーネルになってます。


#kernel1.norm = "ornstein process"
numofspan = 100 #離散化する数 増やせばなめらかさは増しますが、計算コストも増えます。
gram_matrix = np.identity(numofspan)
x_range = np.linspace(-1,1,numofspan)
for i in range(numofspan):
    for k in range(numofspan):
        x1 = x_range[i]
        x2 = x_range[k]
        gram_matrix[i][k] = kernel1.kernel(np.array([x1]),np.array([x2]))
#plt.plot(range(len(gram_matrix[0])),np.abs(np.linalg.eig(gram_matrix)[0]))
#plt.show()
"""
    #グラム行列の固有値がどうなってるか調べたかったので。
    #本来はグラム行列が正定値である条件を満たしていないと、カーネルの条件に反してしまうので
    #分布を作れないんですが、np.random.multivariate_normalは勝手に疑似逆行列を生成してくれるみたいで
    #（正定値じゃないぞコラ）というエラーを吐きながらちゃんとサンプリングしてくれます。えらい！
    """

color = ["g","r","b","y"]
for i in range(10):
    y = np.random.multivariate_normal(np.zeros(numofspan),gram_matrix,1)
    plt.plot(x_range,y[0],color[i % len(color)])
plt.show()
