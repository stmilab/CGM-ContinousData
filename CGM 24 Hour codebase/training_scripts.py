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
    activity_encoder, cgm_encoder, meal_time_encoder, image_encoder,image_set_model, regressor, 
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
            list(image_set_model.parameters()),
            lr=lr
        )
    else:

        optimizer = optim.Adam(
            list(activity_encoder.parameters()) +
            list(cgm_encoder.parameters()) +
            list(meal_time_encoder.parameters()) +
            list(image_encoder.parameters()) +
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
        
        epoch_loss = 0.0
        for batch in train_loader:
            # Extract and move data to respective devices
            activity_data = batch["activity_data"].to(device_activity)
            cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
            meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
            demographics = batch['demographics'].to(device_regressor)
            
            # Process image data using the dedicated function
            image_embs = process_image_embeddings_with_transformer(batch["images"], image_encoder,image_set_model, device_image, device_image)
            #image_embs = process_image_embeddings(batch["images"], image_encoder, device_image, device_regressor)

            # Process labels
            labels = process_labels(batch["nutrition"], global_mean, global_std, device_regressor)
            labels = labels.float().to(device_regressor)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # Generate embeddings from each encoder
                    activity_emb = activity_encoder(activity_data)
                    cgm_emb = cgm_encoder(cgm_data)
                    meal_time_emb = meal_time_encoder(meal_timing_data)
                    
                    # Move embeddings to regressor device and ensure consistent shape
                    activity_emb = activity_emb.to(device_regressor)
                    cgm_emb = cgm_emb.to(device_regressor)
                    meal_time_emb = meal_time_emb.to(device_regressor)
                    image_embs = image_embs.to(device_regressor)
                    # Ensure all embeddings have the same first dimension (batch_size)
                    batch_size = activity_emb.size(0)
                    assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                    assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                    assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                    assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                    
                    # Concatenate all embeddings with demographics
                    joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs, demographics], dim=1).to(torch.float32)
                    
                    # Final prediction
                    pred = regressor(joint_emb).squeeze(1)
                   
                    # Compute loss
                    loss = criterion(pred, labels).mean(dim=0)
                
                # Backpropagation with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard processing without mixed precision
                activity_emb = activity_encoder(activity_data)
                cgm_emb = cgm_encoder(cgm_data)
                meal_time_emb = meal_time_encoder(meal_timing_data)
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs, demographics], dim=1).to(torch.float32)
                
                # Final prediction
                pred = regressor(joint_emb)
                
                # Compute loss
                loss = criterion(pred, labels).mean(dim=0)
                loss.backward()
                optimizer.step()
            
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
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Extract and move data to respective devices
                activity_data = batch["activity_data"].to(device_activity)
                cgm_data = batch['cgm_data'][:, 0, :].unsqueeze(1).to(device_cgm)
                meal_timing_data = torch.stack(batch['meal_timing_features'], dim=0).to(device_meal)
                demographics = batch['demographics'].to(device_regressor)
                
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
                
                # Move embeddings to regressor device and ensure consistent shape
                activity_emb = activity_emb.to(device_regressor)
                cgm_emb = cgm_emb.to(device_regressor)
                meal_time_emb = meal_time_emb.to(device_regressor)
                image_embs = image_embs.to(device_regressor)
                # Ensure all embeddings have the same first dimension (batch_size)
                batch_size = activity_emb.size(0)
                assert cgm_emb.size(0) == batch_size, "CGM embedding batch dimension mismatch"
                assert meal_time_emb.size(0) == batch_size, "Meal timing embedding batch dimension mismatch"
                assert image_embs.size(0) == batch_size, "Image embedding batch dimension mismatch"
                assert demographics.size(0) == batch_size, "Demographics batch dimension mismatch"
                
                # Concatenate all embeddings with demographics
                joint_emb = torch.cat([cgm_emb, activity_emb, meal_time_emb, image_embs, demographics], dim=1).to(torch.float32)
                
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
