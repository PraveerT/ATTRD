# Dual Model Training with Very Late Fusion

This implementation provides a dual model training approach that keeps two separate models (motion and spatial attention) and performs very late fusion at the prediction level.

## Models

1. **Motion Model** (`models/motion.py`): The original temporal motion model that focuses on temporal relationships in point cloud sequences.

2. **Spatial Attention Model** (`models/spatial_attention.py`): A new model that focuses on spatial relationships and patterns in point clouds.

3. **Dual Model Trainer** (`models/dual_model.py`): A trainer that combines both models with very late fusion.

## Key Features

- **Separate Training**: Both models are trained together but maintain their separate architectures.
- **Very Late Fusion**: Fusion occurs after individual predictions, not during feature extraction.
- **Individual Accuracies**: Each model's accuracy is calculated and reported separately.
- **Adaptive Fusion Weights**: Learnable weights determine how much each model contributes to the final prediction.

## Configuration

The dual model uses the configuration file `dual_model.yaml` which specifies:
- The dual model trainer as the main model
- Shared parameters for both models
- Separate training and evaluation procedures

## Usage

To train the dual model:

```bash
cd experiments
python main.py --config dual_model.yaml
```

Or use the provided script:

```bash
cd experiments
./train_dual_model.sh
```

## Architecture Details

### Motion Model
- Uses Mamba-based temporal encoder for sequence modeling
- Multi-scale feature processing
- Quaternion transformations for rotation-equivariant features

### Spatial Attention Model
- Focuses on spatial relationships between points
- Uses spatial attention mechanisms to weight important features
- Enhanced spatial feature processing with convolutional layers

### Dual Model Trainer
- Keeps both models separate during training
- Combines predictions using learnable fusion weights
- Provides individual accuracy metrics for each model
- Implements true very late fusion at the prediction level

## Benefits

1. **Model Specialization**: Each model can specialize in its domain (temporal vs spatial)
2. **Flexibility**: Models can be used independently or together
3. **Transparency**: Individual model performance is clearly visible
4. **Robustness**: Combining different approaches can improve overall performance