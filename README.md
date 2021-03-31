# Weakly supervised semantic segmentation - pytorch

This is a mini research project investigating the effect of weak supervision on a semantic segmentation problem on an aerial imagery dataset.

## Method

This project includes pytorch implementation of the UNet model, with additional weakly supervised losses.

loss_weak - Loss based on comparing class predictions for the whole crop.

loss_flip - Loss based on comparing model outputs of flipped/non-flipped input images.

## Data

Data used here is the ISPRS Vaihingen dataset. You can get it from their website:
http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html

## TODOs

* Evaluation script that computes losses and MAP on test set per each class
* Add general data augmentation (flip, rotate, etc.)
* Inference script that takes full image as input, splits into chunks, runs through the model, and stiches it all back together
