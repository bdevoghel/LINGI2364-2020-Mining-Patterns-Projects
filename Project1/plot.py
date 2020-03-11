import matplotlib.pyplot as plt

fig = plt.figure()
plt.subplot(221)
plt.plot([90,80,70,60,50,40,30,20,10], [1.04,1.03,1.02,1.02,1.03,1.10,1.11,1.10,1.65], [90,80,70,60,50,40,30,20,10], [1.56,1.48,1.49,1.47,1.57,1.66,1.65,1.67,1.83])
plt.ylabel("Total time (s)")
plt.title("retail.dat")
mush = fig.add_subplot(2, 2, 2)
mush.plot([90,80,70,60,50,40,30,20], [0.24,0.50,0.47,1.19,4.98,26.15,99.92,1675.65], label="Apriori")
mush.plot([90,80,70,60,50,40,30,20,10], [0.21,0.23,0.34,0.26,0.42,0.86,2.63,31.66,299.16], label="Alternative Miner")
mush.set_yscale('log')
mush.legend()
plt.subplot(222)
plt.title("mushroom.dat")
plt.subplot(223)
plt.plot([90,80], [12.91,187.48], [90,80,70,60], [0.61,5.02,27.63,145.64])
plt.ylabel("Total time (s)")
plt.xlabel("Minimum support (%)")
plt.title("chess.dat")
plt.subplot(224)
plt.plot([90,80], [19.72,171.25], [90,80,70,60,50], [18.86,26.17,42.41,106.25,305.03])
plt.xlabel("Minimum support (%)")
plt.title("accidents.dat")

plt.show()