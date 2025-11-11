import matplotlib.pyplot as plt
from IPython.core.display_functions import clear_output

def create_live():
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Live Loss Plot")

def plot(losses, accuracies):
    clear_output(wait=True)

    fig, ax1 = plt.subplots()

    # Plot loss on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, '-o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(accuracies, '-o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Loss and Accuracy Progress")
    plt.savefig('Training_Progress.png')
