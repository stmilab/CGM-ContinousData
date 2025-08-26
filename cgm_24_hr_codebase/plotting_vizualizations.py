import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np

from training_scripts import minute_intensity_batch_to_tensor,process_image_embeddings_with_transformer,process_labels

def compute_modality_saliency_encoder(
    activity_encoder, cgm_encoder, meal_time_encoder, image_encoder, image_set_model, intensity_encoder, regressor,
    sample, global_mean, global_std,
    device_activity="cuda:0", device_cgm="cuda:1", device_meal="cuda:0",
    device_image="cuda:1", device_regressor="cuda:0"
):
    # Set models to devices and eval mode
    activity_encoder = activity_encoder.to(device_activity).eval()
    cgm_encoder = cgm_encoder.to(device_cgm).eval()
    meal_time_encoder = meal_time_encoder.to(device_meal).eval()
    image_encoder = image_encoder.to(device_image).eval()
    image_set_model = image_set_model.to(device_image).eval()
    intensity_encoder = intensity_encoder.to(device_meal).eval()
    regressor = regressor.to(device_regressor).eval()

    # Extract and move input
    activity_data = sample["activity_data"].to(device_activity)
    cgm_data = sample["cgm_data"][:, 0, :].unsqueeze(1).to(device_cgm)
    meal_timing_data = torch.stack(sample["meal_timing_features"], dim=0).to(device_meal)
    intensity = sample['intensity_minute'].to(device_meal)
    demographics = sample['demographics'].to(device_regressor)
    intensity_img = minute_intensity_batch_to_tensor(intensity, device=device_meal)
    image_embs = process_image_embeddings_with_transformer(sample["images"], image_encoder, image_set_model, device_image, device_image)

    # Labels for reference (not needed here)
    labels = process_labels(sample["nutrition"], global_mean, global_std, device_regressor).float().to(device_regressor)

    # Get embeddings
    act_emb = activity_encoder(activity_data).detach().to(device_regressor)
    cgm_emb = cgm_encoder(cgm_data).detach().to(device_regressor)
    meal_emb = meal_time_encoder(meal_timing_data).detach().to(device_regressor)
    intensity_emb = intensity_encoder(intensity_img).detach().to(device_regressor)
    image_embs = image_embs.detach().to(device_regressor)
    demographics = demographics.detach().to(device_regressor)

    # Make all require gradients
    act_emb.requires_grad = True
    cgm_emb.requires_grad = True
    meal_emb.requires_grad = True
    image_embs.requires_grad = True
    intensity_emb.requires_grad = True
    demographics.requires_grad = True

    # Forward through regressor
    joint_emb = torch.cat([cgm_emb, act_emb, meal_emb, image_embs, intensity_emb, demographics], dim=1)
    pred = regressor(joint_emb).squeeze()

    # Ensure scalar before backward
    if pred.numel() > 1:
        pred = pred.mean()

    pred.backward()

    # Compute norms of gradients as importance scores
    modality_scores = {
        "CGM": torch.norm(cgm_emb.grad, dim=1).detach().cpu().numpy(),
        "Activity": torch.norm(act_emb.grad, dim=1).detach().cpu().numpy(),
        "Meal Time": torch.norm(meal_emb.grad, dim=1).detach().cpu().numpy(),
        "Image": torch.norm(image_embs.grad, dim=1).detach().cpu().numpy(),
        "Intensity": torch.norm(intensity_emb.grad, dim=1).detach().cpu().numpy(),
        "Demographics": torch.norm(demographics.grad, dim=1).detach().cpu().numpy(),
    }

    return modality_scores

def compute_modality_saliency(
    activity_encoder, cgm_encoder, meal_time_encoder, image_encoder, image_set_model, intensity_cnn, regressor,
    sample, global_mean, global_std,
    device_activity="cuda:0", device_cgm="cuda:1", device_meal="cuda:0",
    device_image="cuda:1", device_regressor="cuda:0"
):
    # Set to eval mode
    activity_encoder.eval()
    cgm_encoder.eval()
    meal_time_encoder.eval()
    image_encoder.eval()
    regressor.eval()
    intensity_cnn.eval()
    image_set_model.eval()

    # Extract and move input
    activity_data = sample["activity_data"].to(device_activity)
    cgm_data = sample["cgm_data"][:, 0, :].unsqueeze(1).to(device_cgm)
    meal_timing_data = torch.stack(sample["meal_timing_features"], dim=0).to(device_meal)
    intensity = sample['intensity_minute'].to(device_meal)
    demographics = sample['demographics'].to(device_regressor)
    intensity_img = minute_intensity_batch_to_tensor(intensity, device=device_meal)
    image_embs = process_image_embeddings_with_transformer(sample["images"], image_encoder, image_set_model, device_image, device_image)

    # Labels for reference (not needed here)
    labels = process_labels(sample["nutrition"], global_mean, global_std, device_regressor).float().to(device_regressor)

    # Get embeddings
    act_emb = activity_encoder(activity_data).detach().to(device_regressor)
    cgm_emb = cgm_encoder(cgm_data).detach().to(device_regressor)
    meal_emb = meal_time_encoder(meal_timing_data).detach().to(device_regressor)
    intensity_emb = intensity_cnn(intensity_img).detach().to(device_regressor)
    image_embs = image_embs.detach().to(device_regressor)
    demographics = demographics.detach().to(device_regressor)

    # Make all require gradients
    act_emb.requires_grad = True
    cgm_emb.requires_grad = True
    meal_emb.requires_grad = True
    image_embs.requires_grad = True
    intensity_emb.requires_grad = True
    demographics.requires_grad = True

    # Forward through regressor
    joint_emb = torch.cat([cgm_emb, act_emb, meal_emb, image_embs, intensity_emb, demographics], dim=1)
    pred = regressor(joint_emb).squeeze()

    # Ensure scalar before backward
    if pred.numel() > 1:
        pred = pred.mean()

    pred.backward()

    # Compute norms of gradients as importance scores
    modality_scores = {
        "CGM": torch.norm(cgm_emb.grad, dim=1).detach().cpu().numpy(),
        "Activity": torch.norm(act_emb.grad, dim=1).detach().cpu().numpy(),
        "Meal Time": torch.norm(meal_emb.grad, dim=1).detach().cpu().numpy(),
        "Image": torch.norm(image_embs.grad, dim=1).detach().cpu().numpy(),
        "Intensity": torch.norm(intensity_emb.grad, dim=1).detach().cpu().numpy(),
        "Demographics": torch.norm(demographics.grad, dim=1).detach().cpu().numpy(),
    }

    return modality_scores


def compute_modality_saliency_unified(
    unified_encoder, image_encoder, image_set_model, regressor,
    batch, global_mean, global_std,
    device_timeseries="cuda:0", device_image="cuda:1", device_regressor="cuda:0"
):
    def extract_sample_from_batch(batch, index):
        sample = {
            key: batch[key][index].unsqueeze(0) if isinstance(batch[key], torch.Tensor) else [batch[key][index]]
            for key in batch
        }
        return sample
    
    sample = extract_sample_from_batch(batch,1)
    # Set models to evaluation mode
    unified_encoder.eval()
    image_encoder.eval()
    image_set_model.eval()
    regressor.eval()

    # Move models to appropriate devices
    unified_encoder.to(device_timeseries)
    image_encoder.to(device_image)
    image_set_model.to(device_image)
    regressor.to(device_regressor)

    # Extract and move time series data
    activity_data = sample["activity_data"].to(device_timeseries)         # [1, 2, 1440]
    cgm_data = sample["cgm_data"][:, 0, :].unsqueeze(1).to(device_timeseries)  # [1, 1, 1440]
    meal_timing_data = torch.stack(sample["meal_timing_features"], dim=0).to(device_timeseries)  # [1, 5, 1440]
    intensity = sample["intensity_minute"].unsqueeze(1).to(device_timeseries)  # [1, 1, 1440]

    # Unified input: [1, 9, 1440]
    unified_input = torch.cat([activity_data, cgm_data, meal_timing_data, intensity], dim=1)
    unified_input.requires_grad = True
    unified_input.retain_grad()  # Ensure gradients are retained for non-leaf tensor

    # Process image embeddings
    image_embs = process_image_embeddings_with_transformer(
        sample["images"], image_encoder, image_set_model, device_image, device_image
    ).detach().to(device_regressor)

    demographics = sample["demographics"].to(device_regressor)

    # Normalize and move labels
    labels = process_labels(sample["nutrition"], global_mean, global_std, device_regressor)
    labels = labels.float().to(device_regressor)

    # Forward pass
    unified_emb = unified_encoder(unified_input).to(device_regressor)
    joint_emb = torch.cat([unified_emb, image_embs, demographics], dim=1).to(torch.float32)
    pred = regressor(joint_emb).squeeze()

    if pred.numel() > 1:
        pred = pred.mean()

    pred.backward()

    # Gradient from unified input: [1, 9, 1440]
    grad = unified_input.grad.detach().cpu()

    # Slicing modalities from channel axis
    modality_slices = {
        "Activity": slice(0, 2),       # channels 0,1
        "CGM": slice(2, 3),            # channel 2
        "Meal Time": slice(3, 8),      # channels 3–7
        "Intensity": slice(8, 9)       # channel 8
    }

    modality_scores = {}
    for name, ch_slice in modality_slices.items():
        modality_grad = grad[:, ch_slice, :]  # [1, ch, T]
        norm = torch.norm(modality_grad, dim=(1, 2)).mean().item()  # mean over batch
        modality_scores[name] = norm

    return modality_scores



def plot_modality_saliency(modality_scores, filename="saliency_map.png"):
    # Compute absolute mean gradient norms
   
    modality_mean_scores = {k: np.abs(v).mean() for k, v in modality_scores.items()}

    # Sort modalities by descending importance
    sorted_modalities = sorted(modality_mean_scores.items(), key=lambda x: x[1], reverse=True)
    modalities, scores = zip(*sorted_modalities)

    # Create plot
    plt.figure(figsize=(8, 4))
    plt.bar(modalities, scores, color="salmon")
    plt.ylabel("Mean |Gradient| (Importance)")
    plt.title("Saliency per Modality Based on Regressor Output")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Ensure directory exists and save
    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def plot_loss_curves(
    training_losses, validation_losses,
    xlabel="Epoch", ylabel="Loss",
    title="Training and Validation Loss",
    filename="training result images/loss_plot.png"
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to '{filename}'")


def plot_pearson_correlation(
    correlations,
    xlabel="Epoch", ylabel="Pearson r",
    title="Validation Correlation",
    filename="training result images/correlation_plot.png"
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure()
    plt.plot(correlations, label='Pearson Correlation', color='green')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Correlation plot saved to '{filename}'")