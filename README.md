# Land Mine Classification using Deep Learning Models

## This repository contains the implementation of a Land Mine Classification project using deep learning models.

### Dataset
- The dataset used for this project is from Kaggle: Land Mine Classification Dataset.

### Models Used
- The following deep learning models were implemented and analyzed:

1. CNN (Convolutional Neural Network)
- Input Layer: Grayscale image (28×28×1).
- Convolutional Layer 1: 5×5 kernel with valid padding, producing (24×24×n1) feature maps.
- Max-Pooling Layer 1: 2×2 pooling, reducing size to (12×12×n1).
- Convolutional Layer 2: 5×5 convolution, resulting in (8×8×n2).
- Max-Pooling Layer 2: 2×2 pooling, reducing size to (4×4×n2).
- Fully Connected Layers:
- fc_3: ReLU activated layer with n3 units.
- fc_4: Dropout layer for classification.
- Output Layer: Softmax layer for class predictions.

2. VGGNet
- Input Layer: Image of size (224×224×3).
- Convolutional Layers: Stacks of 3×3 convolution filters with ReLU activation:
- conv1: 64 filters
- conv2: 128 filters
- conv3: 256 filters
- conv4: 512 filters
- conv5: 512 filters
- Max Pooling: After each convolution block for spatial reduction.
- Fully Connected Layers:
- fc6 and fc7: 4096 units (ReLU activated)
- fc8: 1000 units for classification (ImageNet-style output)
- Output Layer: Softmax for multi-class classification.

3. Inception V3
- Inception Modules: Parallel 1×1 and 3×3 convolutions capturing multi-scale features.
- Pooling Layers: Max and Mean pooling for down-sampling.
- Concatenation Layer: Merges multi-scale features from Inception modules.
- Fully Connected Layer: Dense layers with Dropout for regularization.
- Output Layer: Softmax for classification.
- Optimized for accuracy and efficiency, suitable for real-world detection tasks.

4. ResNet101 (Residual Network)
- Residual Blocks: Identity shortcuts with 3-layer bottleneck blocks (1×1, 3×3, 1×1 convolutions).
- Layer Stacks:
- Conv2_x: 256 filters ×3 repeats
- Conv3_x: 512 filters ×4 repeats
- Conv4_x: 1024 filters ×23 repeats
- Conv5_x: 2048 filters ×3 repeats
- Pooling: Average pooling to reduce dimensions to 1×1.
- Fully Connected Layer: Final classification layer.
- Skip connections help avoid vanishing gradients, enabling deep architectures with improved accuracy.

### Dependencies
- Python 3.10+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
