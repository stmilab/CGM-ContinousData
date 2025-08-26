from data_loader import process_multiple_subjects, get_train_test_datasets
from utils import filter_samples_by_nutrition, custom_collate
from torch.utils.data import DataLoader
from Models import MultiheadAttention as TransformerEncoder,ImprovedRegressor, ImageEncoder, ImageSetTransformer,MultiChannelTransformerEncoder,Regressor,IntensityCNN
from training_scripts import train_model_all_modalities,train_model_all_modalities_cnn_heatmap,train_model_unified_timeseries
import matplotlib.pyplot as plt
from plotting_vizualizations import compute_modality_saliency_unified,compute_modality_saliency,plot_modality_saliency,plot_loss_curves,plot_pearson_correlation,compute_modality_saliency_encoder

# summary = process_multiple_subjects(
#     subject_ids=range(1, 50),  # Process subjects 1-49
#     csv_dir="CGMacros-2",  # Path to your CSV files
#     save_dir="processed_data/",  # Where to save processed data
#     cgm_cols=["Dexcom GL", "Libre GL"],
#     activity_cols=["HR","METs"],
#     img_size=(112, 112),
#     start_hour=6  # New parameter: starting hour (6 AM)
# )

# print(f"Processing summary:")
# print(f"- Processed {len(summary['processed_subjects'])} subjects")
# print(f"- Total days: {summary['total_days']}")
# print(f"- Total images: {summary['total_images']}")
# print(f"- Total meals: {summary['total_meals']}")  # Added meal count

#Step 2: Create train and test datasets from the processed data
train_dataset, test_dataset = get_train_test_datasets(
    data_dir="processed_data/",
    subject_ids=None,  # Use all available subjects
    test_size=0.2,  # 80% train, 20% test
    random_state=2025,  # For reproducibility
    transform=None  # Add any transforms you need
)

print("Done creating train and test datasets")



#Step 3: Create DataLoaders for efficient batching
train_loader = DataLoader(
    filter_samples_by_nutrition(train_dataset),
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate
)

test_loader = DataLoader(
    filter_samples_by_nutrition(test_dataset),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate
)

image_encoder = ImageEncoder(
    image_size=224,  # Standard input size for many vision models
    patch_size=16,   # Typical patch size for ViT
    num_classes=32,  # Match output dimension with other encoders
    channels=3,      # RGB images
    dropout=0.2      # Consistent with other components
    
)

activity_encoder = MultiChannelTransformerEncoder(
    n_features=1440,  # Your sequence length
    n_channels=2,     # Number of channels in your tensor
    embed_dim=96,
    num_heads=2,
    num_classes=32,
    dropout=0.2,
    num_layers=3
)
cgm_encoder = TransformerEncoder(n_features=1440, embed_dim=96, num_heads=2, num_classes=32, dropout=0.2, num_layers=3)
intensity_encoder = MultiChannelTransformerEncoder(
    n_features=1440,  # Your sequence length
    n_channels=1,     # Number of channels in your tensor
    embed_dim=96,
    num_heads=2,
    num_classes=32,
    dropout=0.2,
    num_layers=3
)


intensityCnn = IntensityCNN(output_dim=32)
meal_time_encoder = MultiChannelTransformerEncoder(
    n_features=1440,  # Your sequence length
    n_channels=5,     # Number of channels in your tensor
    embed_dim=96,
    num_heads=2,
    num_classes=16,
    dropout=0.2,
    num_layers=3
)
# Initialize regressor
regressor = Regressor(
    input_size=32+32+32+16+5+32,  # Sum of all embedding dimensions plus demographics
    hidden=128,
    output_size=1,  # Adjust based on your nutrition prediction targets
    dropout=0.2
)
set_transformer = ImageSetTransformer(input_dim=32, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1)

# for batch in train_loader:
#     print(batch['cgm_data'].shape)
#     print(batch['activity_data'].shape)
#     print(batch['intensity_minute'].shape)
#     break
# Create the final model
#training_losses, validation_losses,pearson,_,_ = train_model(activity_encoder,cgm_encoder,meal_time_encoder,regressor,train_loader,test_loader,0,1,epochs=30,lr=1e-4)


unified_encoder = MultiChannelTransformerEncoder(
    n_features=1440,  # Your sequence length
    n_channels=9,     # Number of channels in your tensor
    embed_dim=96,
    num_heads=4,
    num_classes=64,
    dropout=0.2,
    num_layers=3
)


regressor = ImprovedRegressor(
        input_size=64 + 32 + 5,  # unified_embedding + image_embedding + demographics
        hidden_size=256,
        output_size=1,
        dropout=0.3
    )

training_losses, validation_losses, pearson = train_model_unified_timeseries(
    unified_encoder=unified_encoder,
    image_encoder=image_encoder,
    image_set_model=set_transformer,
    regressor=regressor,
    train_loader=train_loader,
    val_loader=test_loader,
    global_mean=0,
    global_std=1,
    epochs=10,
    lr=5e-4,
    weight_decay=1e-7
)



# training_losses, validation_losses, pearson = train_model_all_modalities(
#         activity_encoder=activity_encoder,
#         cgm_encoder=cgm_encoder,
#         meal_time_encoder=meal_time_encoder,
#         image_encoder=image_encoder,
#         image_set_model=set_transformer,
#         intensity_encoder=intensity_encoder,
#         regressor=regressor,
#         train_loader=train_loader,
#         val_loader=test_loader,
#         global_mean=0,
#         global_std=1,
#         epochs=1,
#         lr=1e-4
#     )


# training_losses, validation_losses, pearson = train_model_all_modalities_cnn_heatmap(
#         activity_encoder=activity_encoder,
#         cgm_encoder=cgm_encoder,
#         meal_time_encoder=meal_time_encoder,
#         image_encoder=image_encoder,
#         image_set_model=set_transformer,
#         intensity_cnn = intensityCnn,
#         regressor=regressor,
#         train_loader=train_loader,
#         val_loader=test_loader,
#         global_mean=0,
#         global_std=1,
#         epochs=10,
#         lr=1e-4
#     )



plt.figure(figsize=(12, 5))

# Plot losses

# Create directory if it doesn't exist
plot_loss_curves(training_losses, validation_losses,
                 filename='training result images/loss_allModalities_UnifiedEncoder_plot.png',
                 title="Loss: All Modalities Unified + Intensity Encoder")

plot_pearson_correlation(pearson,
                         filename='training result images/pearson_allModalities_UnifiedEncoder_plot.png',
                         title="Pearson Correlation: All Modalities Unified + Intensity Encoder")


for batch in test_loader:
    break

modality_scores = compute_modality_saliency_unified(
    unified_encoder=unified_encoder,
    image_encoder=image_encoder,
    image_set_model=set_transformer,
    regressor=regressor,
    batch =batch,
    global_mean=0,
    global_std=1
)

plot_modality_saliency(modality_scores, filename="training result images/saliency_allModalitiesUnified_IntensityEncoder_plot.png")