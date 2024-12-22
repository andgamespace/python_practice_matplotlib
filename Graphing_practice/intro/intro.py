import matplotlib.pyplot as plt

plt.plot([1,2,3], [5,7,4])

x = [1, 2, 3]
x2=[1,2,3]
y = [5,7,4]
y2=[]
plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.plot(x, y, label = "First line")
plt.plot(x2, y2, label="Second line")

plt.title('Interesting Graph \n check it out')
plt.legend()
plt.show()