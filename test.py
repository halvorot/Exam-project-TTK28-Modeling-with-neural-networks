
# %%
from keras.models import load_model
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
import numpy as np

# %%
model_name = "alexnet"
model = load_model("Models/{}/model_20_epochs".format(model_name))

# Predicting with the test data
y_test = pd.read_csv("Test.csv")

labels = y_test['ClassId'].values
imgs = y_test['Path'].values

data=[]
if model_name == "lenet5":
    height = 28
    width = 28
elif model_name == "alexnet":
    height = 227
    width = 227
else:
    height = 30
    width = 30

n_channels  = 3
n_classes   = 43

for img in imgs:
    image = Image.open(img)
    image = image.resize((width,height))
    data.append(np.array(image))

X_test = np.array(data)

# %%
pred = model.predict_classes(X_test)

print(model_name, "accuracy: ", accuracy_score(labels, pred))
# %%
