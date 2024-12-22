import matplotlib.pyplot as plt


days=[1,2,3,4,5]

sleeping=[7,8,6,8,9]
eating=[2,3,2,3,4]
working=[6,4,7,5,8]
playing=[4,6,7,4,6]

slices = [7, 2, 2, 13]
activities = ['sleeping', 'eating', 'working', 'playing']
cols = ['c','m','r','b']
plt.pie(slices, 
        labels=activities, 
        colors=cols, 
        startangle=90,
        shadow=True,
        explode=(0,0.1,0,0),
        autopct='%1.1f%%')
        #adds percentages to the pie
#startangle rotates the start of the pie chart by angle degrees counterclockwise from the x-axis.

plt.xlabel('x')
plt.ylabel('y')
plt.title('interesting graph\nCheck it out')
plt.legend()
plt.show()
