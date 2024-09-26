import numpy as np
import matplotlib.pyplot as plt

# x1 = np.linspace(0, 2*np.pi)    # 50x1 array between 0 and 2*pi
# y1 = np.cos(x1)
# x2 = np.linspace(0, 2*np.pi,20) # 20x1 array
# y2 = np.sin(x2)
# x3 = np.linspace(0, 2*np.pi,10) # 10x1 array
# y3 = np.sinc(x3)

with open('sta.txt', 'r') as file:
    # 读取所有行并去掉换行符，返回列表
    lines = file.read().splitlines()

scores = np.array([float(line) for line in lines])
# print(scores)
# episodes = np.array([ for line in lines])

# scores = []
# episodes = []
# i = 1000
# for line in lines:
#     # print(line)
#     scores.append(int(line))
#     episodes.append(i)
#     i += 1000

"""
Taipei_temp = [23.2, 23.4, 23.3, 22.7, 23.2, 23.4, 23.5, 23.8, 24, 23.9]
year = range(2008, 2018)
plt.plot(year, Taipei_temp)
plt.show()
"""

#Example of adding title, axies labels, and text string
plt.plot(scores)
plt.xlabel('episodes (K)')
plt.ylabel('score')
# plt.title(r'sinc(x) between 0 and 2$\pi$')
# plt.text(3,0.5,'example',fontsize=18)
plt.savefig("2048.jpg")
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # 隨機生成 1000 個數據點
# data = np.random.rand(1000)

# # 繪製數據
# plt.plot(data, label="Random Data")

# # 添加標題和標籤
# plt.title("Randomly Generated Data Plot")
# plt.xlabel("Index")
# plt.ylabel("Value")

# # 顯示圖例
# plt.legend()

# # 顯示圖表
# plt.show()

