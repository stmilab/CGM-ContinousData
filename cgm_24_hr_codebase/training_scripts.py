# Currently this model trains for all modalities using the concatenation and attention mechanism for the images defined by the process_image_embeddings_with_transformer method. This can be swapped out 
import pandas as pd
import os, pdb, copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from macro_utils import process_labels,process_labels_lunch_only,RMSRELoss
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


def minute_intensity_batch_to_tensor(batch_intensity, img_size=128, device='cpu'):
    """
    Converts a batch of 1D intensity arrays (shape: B x 1440) into heatmap image tensors.
    Returns a tensor of shape (B, 3, img_size, img_size) on the specified device.
    """
    batch_intensity = batch_intensity.detach().cpu().numpy()  # move to CPU and convert to numpy
    tensors = []

    for intensity_array in batch_intensity:
        fig, ax = plt.subplots(figsize=(2, 2))
        canvas = FigureCanvas(fig)

        reshaped = intensity_array.reshape(24, 60)  # 24 hours × 60 minutes
        ax.imshow(reshaped, cmap='viridis')
        ax.axis('off')
        canvas.draw()

        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha
        plt.close(fig)

        img = Image.fromarray(img).resize((img_size, img_size))
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        tensors.append(tensor)

    result = torch.stack(tensors)  # (B, 3, H, W)
    return result.to(device)  # move result to original device




def process_image_embeddings(batch_images, image_encoder, device_image, device_regressor):
    """
    Process batch of images and generate embeddings with consistent shape.
    Handles both PyTorch tensors and NumPy arrays as input.
    """
    batch_image_embeddings = []

    for sample_images in batch_images:
        if len(sample_images) > 0:
            # Convert NumPy arrays to PyTorch tensors and handle formatting
            sample_image_tensors = []
            for img in sample_images:
                # Convert to tensor
                img_tensor = torch.tensor(img, dtype=torch.float32)
                
                # Ensure image is in channel-first format (C×H×W)
                if img_tensor.shape[-1] == 3:  # If channel is last dimension (H×W×C)
                    img_tensor = img_tensor.permute(2, 0, 1)  # Convert to C×H×W
                
                # Move to device
                img_tensor = img_tensor.to(device_image)
                sample_image_tensors.append(img_tensor)
            
            # Process each image and get embeddings
            sample_image_embs = [image_encoder(img.unsqueeze(0)) for img in sample_image_tensors]
            
            # Average all image embeddings for this sample
            if sample_image_embs:
                avg_image_emb = torch.mean(torch.stack(sample_image_embs), dim=0)
            else:
                avg_image_emb = torch.zeros(image_encoder.output_dim, device=device_image)
        else:
            # No images for this sample, use zero embedding
            avg_image_emb = torch.zeros(image_encoder.output_dim, device=device_image)
        
        # Add this sample's embedding to the batch
        batch_image_embeddings.append(avg_image_emb)
    
    # Stack all image embeddings for the batch
    image_embs = torch.stack(batch_image_embeddings)
    
    # Remove any extra dimensions that might be causing shape issues
    if len(image_embs.shape) > 2:
        image_embs = image_embs.squeeze(1)  # Remove any singleton dimensions
        
    image_embs = image_embs.to(device_regressor)
    
    return image_embs


def process_image_embeddings_with_transformer(image_lists, image_encoder, set_transformer, device_image, device_regressor):
    """
    Encodes variable-length image sets per sample using image_encoder (ViT) followed by set_transformer.

    Args:
        image_lists: List of lists of images per sample. Each inner list contains image tensors [3×H×W].
        image_encoder: The per-image encoder (e.g., ViT).
        set_transformer: The transformer to fuse image embeddings.
        device_image: Device for image encoder.
        device_regressor: Device for set transformer output.
        
    Returns:
        image_set_embeddings: Tensor of shape [batch_size, set_transformer.output_dim]
    """


    batch_size = len(image_lists)

    # Get max number of images across samples
    max_num_images = max(len(images) for images in image_lists)

    # Encode all images, pad to max_num_images
    image_emb_list = []
    padding_mask = []

    for images in image_lists:
        emb_list = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.shape[-1] == 3:  # [H, W, 3] -> [3, H, W]
                    img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img)
            img = img.float() / 255.0  # Normalize if needed
            img = img.to(device_image).unsqueeze(0)  # [1, 3, H, W]

            with torch.no_grad():
                emb = image_encoder(img)  # [1, D]
            emb_list.append(emb.squeeze(0))

        num_imgs = len(emb_list)
        if num_imgs < max_num_images:
            pad = [torch.zeros_like(emb_list[0]) for _ in range(max_num_images - num_imgs)]
            emb_list += pad

        image_emb_list.append(torch.stack(emb_list))  # [max_num_images, D]
        mask = [False] * num_imgs + [True] * (max_num_images - num_imgs)
        padding_mask.append(torch.tensor(mask, dtype=torch.bool))

    # Stack to form batch
    image_embs = torch.stack(image_emb_list).to(device_image)  # [B, S, D]
    padding_mask = torch.stack(padding_mask).to(device_image)  # [B, S]

    # Pass through transformer
    image_set_emb = set_transformer(image_embs, mask=padding_mask)  # [B, D]

    return image_set_emb



def train_model_all_modalities(
    activity_encoder, cgm_encoder, meal_time_encoder, image_encoder,image_set_model,intensity_encoder, regressor, 
    train_loader, val_loader, global_mean, global_std,
    device_activity="cuda:0", device_cgm="cuda:1", device_meal="cuda:0", 
    device_image="cuda:1", device_regressor="cuda:0",
    epochs=30, lr=5e-4, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a model with multiple encoders including image processing and demographics integration at regressor level.
    
    Args:
        activity_encoder: Encoder for activity data
        cgm_encoder: Encoder for CGM data
        meal_time_encoder: Encoder for meal timing features
        image_encoder: Encoder for food images
        regressor: Final regressor that combines all features
        train_loader, val_loader: Data loaders
        global_mean, global_std: Normalization parameters
        device_*: Device assignments for different components
        epochs: Number of training epochs
        lr: Learning rate
        device: Default device if specific devices not available
    """
    # Check device availability and set device configuration
    if not torch.cuda.is_available() and "cuda" in (device_activity, device_cgm, device_meal, device_image, device_regressor):
        print("CUDA not available. Falling back to CPU.")
        device_activity = device_cgm = device_meal = device_image = device_regressor = "cpu"
    elif torch.cuda.device_count() == 1 and any(d != "cuda:0" for d in [device_activity, device_cgm, device_meal, device_image, device_regressor]):
        print(f"Only one CUDA device available. Using cuda:0 for all components.")
        device_activity = device_cgm = device_meal = device_image = device_regressor = "cuda:0"
    
    # Move models to respective devices
    activity_encoder.to(device_activity)
    cgm_encoder.to(device_cgm)
    meal_time_encoder.to(device_meal)
    image_encoder.to(device_image)
    regressor.to(device_regressor)
    image_set_model.to(device_image)
    intensity_encoder.to(device_meal)
    
    # Loss function
    criterion = RMSRELoss()
    if image_set_model:
    # Create a single optimizer for all parameters
        optimizer = optim.Adam(
            list(activity_encoder.parameters()) +
            list(cgm_encoder.parameters()) +
            list(meal_time_encoder.parameters()) +
            list(image_encoder.parameters()) +
            list(regressor.parameters())+
            list(image_set_model.parameters())+
            list(intensity_encoder.parameters()),
            lr=lr
        )
    else:

        optimizer = optim.Adam(
            list(activity_encoder.parameters()) +
            list(cgm_encoder.parameters()) +
            list(meal_time_encoder.parameters()) +
            list(image_encoder.parameters()) +
            list(intensity_encoder.parameters())+
            list(regressor.parameters()), 
            lr=lr
        )
    
    
    # Store loss values
    training_losses, validation_losses, pearson_correlations = [], [], []
    
    # Determine if we can use mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    for epoch in tqdm(range(epochs), ascii=True, desc="Training Epochs"):
        # Set models to training mode
        activity_encoder.train()
        cgm_encoder.train()
        meal_time_encoder.train()
        image_encoder.train()
        regressor.train()
        intensity_encoder.train()
        
        epoch_loss = 0.0
        for batch in train_loader:
            # Extract and move data to respective devices
            activity_data = batch["activity_data"].to(device_activity)
            cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
            meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
            demographics = batch['demographics'].to(device_regressor)
            intensity = batch['intensity_minute'].unsqueeze(1).to(device_meal)
            
            # Process image data using the dedicated function
            image_embs = process_image_embeddings_with_transformer(batch["images"], image_encoder,image_set_model, device_image, device_image)
            #image_embs = process_image_embeddings(batch["images"], image_encoder, device_image, device_regressor)

            # Process labels
            labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
            labels = labels.float().to(device_regressor)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            
            with torch.amp.autocast('cuda'):
                # Generate embeddings from each encoder
                activity_emb = activity_encoder(activity_data)
                cgm_emb = cgm_encoder(cgm_data)
                meal_time_emb = meal_time_encoder(meal_timing_data)
                intensity_emb = intensity_encoder(intensity)
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                intensity_emb = intensity_emb.to(device_regressor)
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                assert intensity_emb.size(0) == batch_size, "Intensity batch dimension mismatch"
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs,intensity_emb, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb).squeeze(1)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
            
            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        activity_encoder.eval()
        cgm_encoder.eval()
        meal_time_encoder.eval()
        image_encoder.eval()
        regressor.eval()
        intensity_encoder.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Extract and move data to respective devices
                activity_data = batch["activity_data"].to(device_activity)
                cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
                meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
                demographics = batch['demographics'].to(device_regressor)
                intensity = batch['intensity_minute'].unsqueeze(1).to(device_meal)
                
                # Process image data using the dedicated function
                image_embs = process_image_embeddings_with_transformer(batch["images"], image_encoder,image_set_model, device_image, device_image)
                #image_embs = process_image_embeddings(batch["images"], image_encoder, device_image, device_regressor)
                # Process labels
                labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
                labels = labels.float().to(device_regressor)
                
                # Generate embeddings
                activity_emb = activity_encoder(activity_data)
                cgm_emb = cgm_encoder(cgm_data)
                meal_time_emb = meal_time_encoder(meal_timing_data)
                intensity_emb = intensity_encoder(intensity)
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                intensity_emb = intensity_emb.to(device_regressor)
                
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                assert intensity_emb.size(0) == batch_size, "Intensity batch dimension mismatch"

                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs,intensity_emb, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
                val_loss += loss.item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
            all_preds_denorm = np.array(all_preds) * global_std + global_mean
            all_labels_denorm = np.array(all_labels) * global_std + global_mean

            try:
                correlation, p_value = pearsonr(all_preds_denorm.squeeze(), all_labels_denorm.squeeze())
                pearson_correlations.append(correlation)
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: {correlation:.4f} (p={p_value:.4f})")
            except Exception as e:
                print(f"Could not calculate correlation: {str(e)}")
                pearson_correlations.append(0.0)
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: N/A")


    print("Training Complete!")
    return training_losses, validation_losses, pearson_correlations





def train_model_all_modalities_cnn_heatmap(
    activity_encoder, cgm_encoder, meal_time_encoder, image_encoder,image_set_model,intensity_cnn, regressor, 
    train_loader, val_loader, global_mean, global_std,
    device_activity="cuda:0", device_cgm="cuda:1", device_meal="cuda:0", 
    device_image="cuda:1", device_regressor="cuda:0",
    epochs=30, lr=5e-4, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a model with multiple encoders including image processing and demographics integration at regressor level.
    
    Args:
        activity_encoder: Encoder for activity data
        cgm_encoder: Encoder for CGM data
        meal_time_encoder: Encoder for meal timing features
        image_encoder: Encoder for food images
        regressor: Final regressor that combines all features
        train_loader, val_loader: Data loaders
        global_mean, global_std: Normalization parameters
        device_*: Device assignments for different components
        epochs: Number of training epochs
        lr: Learning rate
        device: Default device if specific devices not available
    """
    # Check device availability and set device configuration
    if not torch.cuda.is_available() and "cuda" in (device_activity, device_cgm, device_meal, device_image, device_regressor):
        print("CUDA not available. Falling back to CPU.")
        device_activity = device_cgm = device_meal = device_image = device_regressor = "cpu"
    elif torch.cuda.device_count() == 1 and any(d != "cuda:0" for d in [device_activity, device_cgm, device_meal, device_image, device_regressor]):
        print(f"Only one CUDA device available. Using cuda:0 for all components.")
        device_activity = device_cgm = device_meal = device_image = device_regressor = "cuda:0"
    
    # Move models to respective devices
    activity_encoder.to(device_activity)
    cgm_encoder.to(device_cgm)
    meal_time_encoder.to(device_meal)
    image_encoder.to(device_image)
    regressor.to(device_regressor)
    image_set_model.to(device_image)
    intensity_cnn.to(device_meal)
    
    # Loss function
    criterion = RMSRELoss()
    if image_set_model:
    # Create a single optimizer for all parameters
        optimizer = optim.Adam(
            list(activity_encoder.parameters()) +
            list(cgm_encoder.parameters()) +
            list(meal_time_encoder.parameters()) +
            list(image_encoder.parameters()) +
            list(regressor.parameters())+
            list(image_set_model.parameters())+
            list(intensity_cnn.parameters()),
            lr=lr
        )
    else:

        optimizer = optim.Adam(
            list(activity_encoder.parameters()) +
            list(cgm_encoder.parameters()) +
            list(meal_time_encoder.parameters()) +
            list(image_encoder.parameters()) +
            list(intensity_cnn.parameters())+
            list(regressor.parameters()), 
            lr=lr
        )
    
    
    # Store loss values
    training_losses, validation_losses, pearson_correlations = [], [], []
    
    # Determine if we can use mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    for epoch in tqdm(range(epochs), ascii=True, desc="Training Epochs"):
        # Set models to training mode
        activity_encoder.train()
        cgm_encoder.train()
        meal_time_encoder.train()
        image_encoder.train()
        regressor.train()
        intensity_cnn.train()
        
        epoch_loss = 0.0
        for batch in train_loader:
            # Extract and move data to respective devices
            activity_data = batch["activity_data"].to(device_activity)
            cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
            meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
            demographics = batch['demographics'].to(device_regressor)
            intensity = batch['intensity_minute'].to(device_meal)

            intensity_img_batch = minute_intensity_batch_to_tensor(intensity, device=device_meal)
            
            # Process image data using the dedicated function
            image_embs = process_image_embeddings_with_transformer(batch["images"], image_encoder,image_set_model, device_image, device_image)
            #image_embs = process_image_embeddings(batch["images"], image_encoder, device_image, device_regressor)

            # Process labels
            labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
            labels = labels.float().to(device_regressor)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            
            with torch.amp.autocast('cuda'):
                # Generate embeddings from each encoder
                activity_emb = activity_encoder(activity_data)
                cgm_emb = cgm_encoder(cgm_data)
                meal_time_emb = meal_time_encoder(meal_timing_data)
                intensity_emb = intensity_cnn(intensity_img_batch)
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                intensity_emb = intensity_emb.to(device_regressor)
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                assert intensity_emb.size(0) == batch_size, "Intensity batch dimension mismatch"
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs,intensity_emb, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb).squeeze(1)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
            
            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        activity_encoder.eval()
        cgm_encoder.eval()
        meal_time_encoder.eval()
        image_encoder.eval()
        regressor.eval()
        intensity_cnn.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Extract and move data to respective devices
                activity_data = batch["activity_data"].to(device_activity)
                cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
                meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
                demographics = batch['demographics'].to(device_regressor)
                intensity = batch['intensity_minute'].to(device_meal)
                intensity_img_batch = minute_intensity_batch_to_tensor(intensity, device=device_meal)
                
                # Process image data using the dedicated function
                image_embs = process_image_embeddings_with_transformer(batch["images"], image_encoder,image_set_model, device_image, device_image)
                #image_embs = process_image_embeddings(batch["images"], image_encoder, device_image, device_regressor)
                # Process labels
                labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
                labels = labels.float().to(device_regressor)
                
                # Generate embeddings
                activity_emb = activity_encoder(activity_data)
                cgm_emb = cgm_encoder(cgm_data)
                meal_time_emb = meal_time_encoder(meal_timing_data)
                intensity_emb = intensity_cnn(intensity_img_batch)
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                intensity_emb = intensity_emb.to(device_regressor)
                
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                assert intensity_emb.size(0) == batch_size, "Intensity batch dimension mismatch"

                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs,intensity_emb, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
                val_loss += loss.item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
            all_preds_denorm = np.array(all_preds) * global_std + global_mean
            all_labels_denorm = np.array(all_labels) * global_std + global_mean

            try:
                correlation, p_value = pearsonr(all_preds_denorm.squeeze(), all_labels_denorm.squeeze())
                pearson_correlations.append(correlation)
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: {correlation:.4f} (p={p_value:.4f})")
            except Exception as e:
                print(f"Could not calculate correlation: {str(e)}")
                pearson_correlations.append(0.0)
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: N/A")


    print("Training Complete!")
    return training_losses, validation_losses, pearson_correlations



def train_model_unified_timeseries(
    unified_encoder, image_encoder, image_set_model, regressor, 
    train_loader, val_loader, global_mean, global_std,
    device_timeseries="cuda:0", device_image="cuda:1", device_regressor="cuda:0", 
    epochs=10, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu",
    weight_decay=1e-5
):
    """
    Train a model with a unified encoder for all time series data and separate image processing.
    
    Args:
        unified_encoder: Combined encoder for all time series data (activity, CGM, meal timing, intensity)
        image_encoder: Encoder for food images
        image_set_model: Transformer for processing multiple images
        regressor: Final regressor that combines all features
        train_loader, val_loader: Data loaders
        global_mean, global_std: Normalization parameters
        device_*: Device assignments for different components
        epochs: Number of training epochs
        lr: Learning rate
        device: Default device if specific devices not available
        weight_decay: L2 regularization parameter to help prevent overfitting
    """
    # Check device availability and set device configuration
    if not torch.cuda.is_available() and "cuda" in (device_timeseries, device_image, device_regressor):
        print("CUDA not available. Falling back to CPU.")
        device_timeseries = device_image = device_regressor = "cpu"
    elif torch.cuda.device_count() == 1 and any(d != "cuda:0" for d in [device_timeseries, device_image, device_regressor]):
        print(f"Only one CUDA device available. Using cuda:0 for all components.")
        device_timeseries = device_image = device_regressor = "cuda:0"
    
    # Move models to respective devices
    unified_encoder.to(device_timeseries)
    image_encoder.to(device_image)
    image_set_model.to(device_image)
    regressor.to(device_regressor)
    
    # Loss function
    criterion = RMSRELoss()
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.Adam(
        list(unified_encoder.parameters()) +
        list(image_encoder.parameters()) +
        list(image_set_model.parameters()) +
        list(regressor.parameters()),
        lr=lr,
        weight_decay=weight_decay  # Add L2 regularization to help with overfitting
    )
    
    # Learning rate scheduler to reduce LR when plateau is reached
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Store metrics
    training_losses, validation_losses, pearson_correlations = [], [], []
    
    # Determine if we can use mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10  # Stop if no improvement for 10 epochs
    
    # Training loop
    for epoch in tqdm(range(epochs), ascii=True, desc="Training Epochs"):
        # Set models to training mode
        unified_encoder.train()
        image_encoder.train()
        image_set_model.train()
        regressor.train()
        
        epoch_loss = 0.0
        for batch in train_loader:
            # Extract data
            activity_data = batch["activity_data"].to(device_timeseries)
            cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_timeseries)
            meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_timeseries)
            intensity = batch['intensity_minute'].unsqueeze(1).to(device_timeseries)
            demographics = batch['demographics'].to(device_regressor)
            
            # Process image data
            image_embs = process_image_embeddings_with_transformer(
                batch["images"], image_encoder, image_set_model, device_image, device_image
            )
            
            # Process labels
            labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
            labels = labels.float().to(device_regressor)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Stack all time series data along channel dimension
                # Ensure all time series have the same sequence length (1440)
                batch_size = activity_data.size(0)
                
                # Prepare unified time series input
                # Format: [batch_size, channels, time_steps]
                # Concatenate all time series features along the channel dimension
                unified_time_series = torch.cat([
                    activity_data,           # [batch_size, 2, 1440]
                    cgm_data,                # [batch_size, 1, 1440]
                    meal_timing_data,        # [batch_size, 5, 1440]
                    intensity                # [batch_size, 1, 1440]
                ], dim=1)  # Result: [batch_size, 9, 1440]
                
                # Generate unified embedding
                unified_emb = unified_encoder(unified_time_series)
                
                # Move embeddings to regressor device
                unified_emb = unified_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([unified_emb, image_embs, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb).squeeze(1)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
            
            # Backpropagation with scaler
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping to help with training stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(unified_encoder.parameters()) + 
                    list(image_encoder.parameters()) + 
                    list(image_set_model.parameters()) + 
                    list(regressor.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(unified_encoder.parameters()) + 
                    list(image_encoder.parameters()) + 
                    list(image_set_model.parameters()) + 
                    list(regressor.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        unified_encoder.eval()
        image_encoder.eval()
        image_set_model.eval()
        regressor.eval()
        
        all_preds, all_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Extract data
                activity_data = batch["activity_data"].to(device_timeseries)
                cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_timeseries)
                meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_timeseries)
                intensity = batch['intensity_minute'].unsqueeze(1).to(device_timeseries)
                demographics = batch['demographics'].to(device_regressor)
                
                # Process image data
                image_embs = process_image_embeddings_with_transformer(
                    batch["images"], image_encoder, image_set_model, device_image, device_image
                )
                
                # Process labels
                labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
                labels = labels.float().to(device_regressor)
                
                # Prepare unified time series input
                unified_time_series = torch.cat([
                    activity_data,
                    cgm_data,
                    meal_timing_data,
                    intensity
                ], dim=1)
                
                # Generate unified embedding
                unified_emb = unified_encoder(unified_time_series)
                
                # Move embeddings to regressor device
                unified_emb = unified_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([unified_emb, image_embs, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb).squeeze(1)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
                val_loss += loss.item()
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model weights
            best_model_weights = {
                'unified_encoder': unified_encoder.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'image_set_model': image_set_model.state_dict(),
                'regressor': regressor.state_dict()
            }
            torch.save(best_model_weights, 'best_model_weights.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Calculate correlation metrics
        all_preds_denorm = np.array(all_preds) * global_std + global_mean
        all_labels_denorm = np.array(all_labels) * global_std + global_mean

        try:
            correlation, p_value = pearsonr(all_preds_denorm.squeeze(), all_labels_denorm.squeeze())
            pearson_correlations.append(correlation)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: {correlation:.4f} (p={p_value:.4f})")
        except Exception as e:
            print(f"Could not calculate correlation: {str(e)}")
            pearson_correlations.append(0.0)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Pearson Correlation: N/A")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement")
            break
    
    # Load best model weights
    try:
        best_weights = torch.load('best_model_weights.pth')
        unified_encoder.load_state_dict(best_weights['unified_encoder'])
        image_encoder.load_state_dict(best_weights['image_encoder'])
        image_set_model.load_state_dict(best_weights['image_set_model'])
        regressor.load_state_dict(best_weights['regressor'])
        print("Loaded best model weights from checkpoint")
    except FileNotFoundError:
        print("No saved model weights found, using final weights")

    print("Training Complete!")
    return training_losses, validation_losses, pearson_correlations