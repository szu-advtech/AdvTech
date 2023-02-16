from GCForest import GCForest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading the data
iris = load_iris()
X = iris.data
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)

gcf = GCForest(shape_1X=4, window=2, tolerance=0.0, mg_flag=False)
gcf.fit(X_tr, y_tr)

pred_X = gcf.predict(X_te)
print(pred_X)

accuracy = accuracy_score(y_true=y_te, y_pred=pred_X)
print('gcForest test accuracy : {}'.format(accuracy))