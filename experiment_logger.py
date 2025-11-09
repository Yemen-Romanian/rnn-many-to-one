import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class TensorboardLogger:
    FIGURE_SIZE = (12, 10)
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{experiment_name}_{current_time}"
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}", flush_secs=5)

    def log(self, message):
        self.writer.add_text('Logs', message)

    def log_loss(self, value, tag, step):
        self.writer.add_scalar(f"Loss/{tag}", value, step)

    def log_model(self, model):
        device = next(model.parameters()).device
        self.writer.add_graph(model, torch.zeros(1, 10, model.gru.input_size).to(device)) # Example input

    def log_plot(self, prediction, target, step):
        fig, ax = plt.subplots(figsize=TensorboardLogger.FIGURE_SIZE)
        ax.plot(target, label='Target')
        ax.plot(prediction, label='Prediction')
        ax.legend()
        self.writer.add_figure('Prediction vs Target', fig, step)
        plt.close(fig)

    def close(self):
        self.writer.close()
