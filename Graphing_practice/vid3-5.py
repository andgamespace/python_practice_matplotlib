import matplotlib.pyplot as plt

# population_ages = [5, 10, 12, 15, 17, 18, 22, 25, 27, 34, 32, 20, 24, 39, 42, 44, 46, 50, 55, 67, 74, 89, 91, 92, 110, 130]
# x = [2,4,6,8,10]
# y = [6,7,8,2,4]
# ids = [x for x in range(len(population_ages))]
# bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
# x2 = [1,3,5,9,11]
# y2 = [7,8,2,4,3]
# plt.bar(x,y,label='Bars1', color='r')
# plt.bar(x2, y2, label="Bars2", color='c')
# plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)
# x=[1,2,3,4,5,6,7,8]
# y=[3,5,5,6,2,5,7,3]
days=[1,2,3,4,5]

sleeping=[7,8,6,8,9]
eating=[2,3,2,3,4]
working=[6,4,7,5,8]
playing=[4,6,7,4,6]

plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Sleeping', linewidth=5)
#when we have a stack plot in matplotlib, can't use labels
#We can get around this by having empty plots
plt.stackplot(days, sleeping, eating, working, playing, colors=['m','c', 'r', 'k'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('interesting graph\nCheck it out')
plt.legend()
plt.show()
