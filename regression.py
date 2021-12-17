import matplotlib.pyplot as plt
import numpy as np
import datetime
import inspect
import re


class problem:
    __solved = False

    def __init__(self, input, output, h, n, featureCount):
        self.__featureCount = featureCount
        self.__h = h
        self.__n = n
        self.__m = len(output)
        self.__alpha = 1
        self.__input = np.array(input)
        self.__output = np.array(output)

    def plotSets(self):
        if (len(self.__input[0]) == 1):
            plt.plot(self.__input[:, 0], self.__output, 'ro')
            plt.ylabel('y axis')
            plt.xlabel('x axis')
            plt.show()
        elif (len(self.__input[0]) == 2):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.__input[:, 0], self.__input[:, 1], self.__output, marker='o')
            ax.set_xlabel('X0')
            ax.set_ylabel('X1')
            ax.set_zlabel('Z')
            plt.show()
        else:
            print("只能绘制二维/三维图形")

    def solve(self):
        
        if (self.__solved):
            print("问题已求解")
            return
        
        def J(th):
            res = 0
            for i in range(self.__m):
                res += (self.__h(th, self.__input[i]) - self.__output[i]) ** 2
            return res / (2 * self.__m)

        def runIter():
            J_normal = J(self.__theta)
            for i in range(self.__n):
                J_b = J_normal
                J_normal = J(self.__theta)
                if (J_normal > J_b):
                    return False
                
                djs = []
                for j in range(self.__featureCount):
                    ep = 0.00001
                    theta_bigger = self.__theta.copy()
                    theta_bigger[j] += ep
                    J_bigger = J(theta_bigger)
                    djs.append((J_bigger - J_normal) / ep)
                
                self.__theta = self.__theta - self.__alpha * np.array(djs)
                
            self.__cost = J_normal
            return True
        
        start = datetime.datetime.now()
        while True:
            self.__theta = np.ones(self.__featureCount)
            if runIter():
                self.__solved = True
                break
            else:
                self.__alpha = self.__alpha / 3
                continue
        end = datetime.datetime.now()
        print("耗时：" + str(end - start))

    def plot(self):
        if not self.__solved:
            print("该问题未求解或求解失败")
        elif (len(self.__input[0]) == 1):
            plt.plot(self.__input[:, 0], self.__output, 'ro')
            X = np.linspace(self.__input.min(axis=0)[0], self.__input.max(
                axis=0)[0], 256, endpoint=True)
            Y = self.__h(self.__theta, [X])
            plt.plot(X, Y)
            plt.ylabel('y axis')
            plt.xlabel('x axis')
            plt.show()
            
        elif (len(self.__input[0]) == 2):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
            ax.scatter(self.__input[:, 0], self.__input[:, 1], self.__output, marker='o')
            
            X = np.linspace(self.__input.min(axis=0)[0], self.__input.max(
                axis=0)[0], 256, endpoint=True)
            Y = np.linspace(self.__input.min(axis=0)[1], self.__input.max(
                axis=0)[1], 256, endpoint=True)
            X, Y = np.meshgrid(X, Y)
            Z = self.__h(self.__theta, [X, Y])

            surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

            ax.set_xlabel('X0')
            ax.set_ylabel('X1')
            ax.set_zlabel('Z')
            plt.show()
        else:
            print("只能绘制二维/三维图形")

    def show(self):
        if (not self.__solved):
            print("该问题未求解或求解失败")
        else:
            lines = inspect.getsource(self.__h)
            ret = lines[re.search('return', lines).span()[1]:-1]
            thName, xName = inspect.getfullargspec(self.__h).args
            for i in range(self.__featureCount):
                val = self.__theta[i]
                ret = ret.replace(thName+"["+str(i)+"]", "(" + str(val)[:5] + ")" if val < 0 else str(val)[:5])
            for i in range(len(self.__input[0])):
                ret = ret.replace(xName+"["+str(i)+"]", xName+str(i))
            print("拟合结果为：h(x) =" + ret)
            print("Theta：" + str(self.__theta))
            print("最终学习率为：" + str(self.__alpha))
            print("代价值为：" + str(self.__cost))
