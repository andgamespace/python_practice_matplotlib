import matplotlib.pyplot as plt
import numpy as np
#part 1
'''
import csv

x = []
y=[]

with open('Python/Graphing_practice/intro/myDoc.txt') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x,y, label='Loaded from file')
'''
x, y = np.loadtxt('Python/Graphing_practice/intro/myDoc.txt', delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file')


plt.xlabel('x')
plt.ylabel('y')
plt.title('interesting graph\nCheck it out')
plt.legend()
plt.show()
