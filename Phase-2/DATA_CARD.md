# Data Card - DermaVision AI

## Dataset Description

The dataset contains skin lesion medical images for binary classification.

## Classes

- Benign
- Malignant

## Modality

Image data.

## Image Formats

JPG, JPEG, PNG.

## Preprocessing Plan

- Convert images to RGB
- Resize images to 224 x 224 pixels
- Convert images to tensors
- Normalize using ImageNet mean and standard deviation

## Augmentation Plan

- Random horizontal flip
- Random rotation
- Brightness and contrast adjustment
- Random resized crop

## Split Strategy

The dataset will be divided into training, validation, and test sets using a stratified split to preserve class distribution.

## Known Issues

- Possible class imbalance
- Different image resolutions
- Lighting variation
- Blurry or noisy images
- Possible duplicate or corrupted samples

## Ethical Note

This dataset is used only for academic and educational deep learning experimentation. The final system is not a replacement for professional medical diagnosis.
