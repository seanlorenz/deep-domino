# Start up H2O
import h2o
h2o.init()

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
