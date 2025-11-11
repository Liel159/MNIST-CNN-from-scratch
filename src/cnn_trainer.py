from Data import import_data
from prediction_logic import *
x_train, y_train, x_dev, y_dev = import_data()

#reshape data
x_train = x_train.T.reshape(-1, 28, 28)
x_dev = x_dev.T.reshape(-1, 28, 28 )

print(x_train.shape, x_dev.shape)

#Initialisation des Listes
losses = []
accuracies = []

# Create a live plot
create_live()

iterations = 300
learning_rate = 0.0075

kernel, w_fc, b_fc, losses = grad_descent(x_train, y_train, iterations,
                                          learning_rate, losses, accuracies)

save_model(kernel, w_fc, b_fc, filename="cnn_model.pkl")

plt.ioff()  # Turn off interactive mode after training
plt.show()  # Show final plot