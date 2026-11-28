# CIFAR 10
- Image size: 32x32
- Number of classes: 10
- Training set: 50000 images, 5000 per class
- Test set: 10000 images, 1000 per class

# Initial results
- Reproducing table one, with one run for user model on 30 epochs, E = 1, epochs = 20 for the expert model: CTA/PTA pairs

|Data|Architecture|m = 0|m = 300|m=500|m=1000|m=1500|
|----|------------|-----|-------|-----|------|------|
|Cifar 10|ResNet 32||89.2/13.0|88.8/57.8|88.9/93.6|87.3/99.3|

- Reproducing Figure 5a for the sinusoidal trigger

After finding optimistic results with these settings, we try to reproduce the paper's results with the same training settings.
Mean PTA differs considerably for small values of $m$, likely from the fact that labels were generated from a single run of step 2 (number of runs and aggregation method not specified in the paper), while the code suggests that different label sets were computed before calculating the one hot flipped vectors


# Questions to investigate
- what if we change the loss function in step two to a cosine ?
- what if we change the goal of step 2 to match the best state of the expert model instead of matching trajectories ?
- what if, from the same flipped labels, we repeat the user training step from different initializations ?