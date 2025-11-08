from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from models import SimpleGRU

def build_model(config):
    model_config = config["model"]
    model_type = model_config["name"]

    if model_type == "SimpleGRU":
        return SimpleGRU(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            output_size=model_config["output_size"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    optim_type = optimizer_config["name"]
    learning_rate = optimizer_config["lr"]

    if optim_type == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optim_type == "SGD":
        momentum = optimizer_config["momentum"]
        return SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")
