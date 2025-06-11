import torch
import torch.nn as nn
def process_labels_lunch_only(labels, mean, std, device, dtype=torch.float32):
    """
    Processes labels by summing the 'calories' of 'lunch' meals for each instance in a batch.

    Args:
        labels (list of list of dicts): A batch of labels, where each instance is a list of meal records.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.
        device (str): Target device ("cuda:0" or "cpu").
        dtype (torch.dtype): Data type (default: float32).

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) with summed lunch calories per instance.
    """
    batch_calories = [
        sum(entry.get("calories", 0) for entry in instance if entry.get("MealType") == "lunch")  
        for instance in labels
    ]
    labels_tensor = torch.tensor(batch_calories, dtype=dtype).to(device)
    normalized_labels = (labels_tensor - mean) / (std + 1e-8)

    return normalized_labels


class RMSRELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon  # Small value to avoid division by zero
    
    def forward(self, pred, target):
        relative_error = (pred - target) / (target + self.epsilon)
        squared_rel_error = relative_error ** 2
        mean_squared_rel_error = torch.mean(squared_rel_error)
        return torch.sqrt(mean_squared_rel_error)
    

def process_labels(labels,mean,std, device, dtype=torch.float32):
    """
    Processes labels by summing the 'calories' field for each instance in a batch.

    Args:
        labels (list of list of dicts): A batch of labels, where each instance is a list of meal records.
        device (str): Target device ("cuda:0" or "cpu").
        dtype (torch.dtype): Data type (default: float16 for mixed precision).

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) with summed calories per instance.
    """
    batch_calories = [
        sum(entry.get("calories", 0) for entry in instance)  
        for instance in labels
    ]
    labels_tensor = torch.tensor(batch_calories, dtype=dtype).to(device)
    normalized_labels = (labels_tensor - mean) / (std + 1e-8)

    return normalized_labels
