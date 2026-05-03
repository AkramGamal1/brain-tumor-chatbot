# What this model does

This chatbot is connected to a brain MRI classifier built by the team in the
parent ML project. The classifier is an **EfficientNet-B0** image-recognition
model that was *fine-tuned* (taught further) to recognize four kinds of brain
MRI views.

## What the model was trained on

- About 7,200 brain MRI images from a public dataset.
- Four labeled categories: glioma, meningioma, pituitary tumor, and no tumor.
- The categories were balanced (similar number of images per class).
- Images were resized to a standard 224×224 pixels and normalized.

## What the model outputs

For any image given to it, the model outputs:

- A **predicted class** — the one of four it thinks is most likely.
- A list of **probabilities** for all four classes, summing to 100%.
- A confidence number for the top prediction.

This chatbot translates those outputs into plain language and adds context. It
**deliberately does not show numeric confidence** to end users — verbal
descriptions ("fairly certain", "moderately confident", "uncertain") are used
instead, to avoid implying more precision than the underlying probability
supports.

## Reported test-set performance

On the held-out test set used by the developers, the model classified about
**94.8%** of test images correctly. That number is impressive for a small
model, but it should not be read as "94.8% chance any given prediction is
right." Performance varies by class and by image, and real-world images can
differ from training images in ways that hurt accuracy.

## What the model is *not*

- Not a diagnostic device.
- Not a substitute for a radiologist or any other clinician.
- Not aware of your medical history, symptoms, or any context outside the
  pixels of one image.
- Not able to recognize anything outside its four-class horizon (see
  "What this model cannot tell you").

This is not medical advice. Please consult a qualified clinician for any
medical decisions.
