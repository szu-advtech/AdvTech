import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('train_log.csv', sep=";")
df_test = pd.read_csv('test_log.csv', sep=";")
# df[['epoch', 'accuracy', 'loss']].plot(
#     x='epoch',
#     xlabel='epoch',
#     ylabel='%',
#     title='train log'
# )
# plt.show()
#

train_acc = df_train['accuracy'].tolist()
train_loss = df_train['loss'].tolist()

test_acc = df_test['accuracy'].tolist()
test_loss = df_test['loss'].tolist()

epochs = range(0, 10)

plt.plot(epochs, train_acc, 'b', label="train_acc", marker="+")
plt.plot(epochs, train_loss, 'bo', label="train_loss", linestyle=":")

plt.plot(epochs, test_acc, 'r', label="test_acc", marker="+")
plt.plot(epochs, test_loss, 'ro', label="test_loss", linestyle=":")
plt.xlabel("epochs")
plt.ylabel("%")

plt.legend()

plt.show()
