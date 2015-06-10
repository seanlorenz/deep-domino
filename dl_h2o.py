# Fix issue with python install location
import sys
sys.prefix = "/usr/local"

# Start up H2O
import h2o
h2o.init(start_h2o=True)

# Load the dataset
prostate = h2o.upload_file(path=h2o.locate("datasets/prostate.csv"))
prostate.describe()

# Set the CAPSULE column to be a factor column then build the model
prostate["CAPSULE"] = prostate["CAPSULE"].asfactor()
model = h2o.deeplearning(x=prostate[list(set(prostate.col_names()) - set(["ID", "CAPSULE"]))],
                         y = prostate["CAPSULE"],
                         training_frame=prostate,
                         activation="Tanh",
                         hidden=[10, 10, 10],
                         epochs=10000)
model.show()

# Make predictions with the trained model
predictions = model.predict(prostate)
predictions.show()

# Check performance of the classification model
performance = model.model_performance(prostate)
performance.show()

# Domino Diagnostic Statistics
r2 = performance.r2()
mse = performance.mse()
auc = performance.auc()
accuracy = performance.metric('accuracy')[0][1]

import json
with open('dominostats.json', 'wb') as f:
    f.write(json.dumps({"R^2": r2, "MSE": mse, "AUC": auc, "Accuracy": accuracy}))