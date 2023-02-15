import matplotlib.pyplot as plt
import numpy as np


texts = ['../C100_{}_results.txt','../C110_{}_results.txt', '../C150_{}_results.txt', '../C200_{}_results.txt', '../C500_{}_results.txt', '../C1000_{}_results.txt']
p = 1

plt.figure(figsize=(15,13), dpi=80)
# each cluster number
for text in texts:
    qe_each_times_average = []
    for i in range(0, 25):
        qe_each_times_average.append(0)

    t = 0
    for times in range(0,10):
        with open(text.format(times)) as f:
            l = f.readlines()
            for i in range(27, 52):
                li = l[i]
                qe = li.split("--")[3].split("\t")[1]
                qe_each_times_average[i-27] += float(qe)
        t += 1

    for i in range(len(qe_each_times_average)):
        qe_each_times_average[i] /= t
    mina = np.min(qe_each_times_average)
    maxa = np.max(qe_each_times_average)

    plt.subplot(6, 1, p)
    plt.xlabel("iteration")
    plt.ylabel("q-error")
    plt.title(text.split("/")[1].split('.')[0].replace("{}", "").replace("_", ""))
    plt.xlim(0, 26)
    plt.ylim(mina, maxa*1.01)
    plt.plot(qe_each_times_average, 'ro-', color='red', linewidth=1, label='Quantization Error')
    p+=1

plt.savefig("q-error-sub.png")

colors = ['#00FFFF','#A52A2A','#B8860B','#9400D3','#FFD700','#20B2AA']
labels = ['100C', '110C', '150C', '200C', '500C', '1000C']
p=1
plt.figure(figsize=(15,10), dpi=80)
plt.xlim(0, 26)
for text in texts:
    qe_each_times_average = []
    for i in range(0, 25):
        qe_each_times_average.append(0)

    t = 0
    for times in range(0,10):
        # get result of each time
        with open(text.format(times)) as f:
            l = f.readlines()
            # get the q-error column
            for i in range(27, 52):
                li = l[i]
                qe = li.split("--")[3].split("\t")[1]
                qe_each_times_average[i-27] += float(qe)
        t += 1

    for i in range(len(qe_each_times_average)):
        qe_each_times_average[i] /= t


    plt.plot(qe_each_times_average, 'ro-', color=colors[p-1], linewidth=1, label=labels[p-1])

    p+=1

plt.xlabel("iteration")
plt.ylabel("q-error")
plt.legend(loc="upper right", title='Cluster Number')
plt.savefig("q-error-one.png")
