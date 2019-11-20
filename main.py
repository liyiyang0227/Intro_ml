import pandas as pd
import numpy as np
from part2_claim_classifier import ClaimClassifier
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("part2_data.csv").values
X = dataset[:,0:9]
Y = dataset[:,-1]
nn = ClaimClassifier()
nn.fit(X,Y)
nn.evaluate_architecture(X,Y)
# nn.save_model()
# print(nn.predict(X))
#model = nn.fit_skl(X,Y)
data_test = dataset[np.where(dataset[:,-1]==1)]
X = data_test[:,0:9]
Y = data_test[:,-1]
y_pred = nn.predict(X)
print(y_pred)
print(accuracy_score(Y,y_pred))
