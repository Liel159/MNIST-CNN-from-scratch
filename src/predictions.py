from prediction_logic import *

kernel, w_fc, b_fc = load_model("cnn_model.pkl")

x_example = load_image("number3.jpg")
prediction = predict(x_example, kernel, w_fc, b_fc)

print(f"The model predicts : {prediction}")