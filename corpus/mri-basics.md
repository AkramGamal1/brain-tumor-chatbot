# MRI basics, in plain language

**MRI** stands for **Magnetic Resonance Imaging**. It is a way of taking
detailed pictures of the inside of the body — including the brain — without
using X-rays or radiation.

## How it works (very high level)

An MRI scanner uses a strong magnet and radio waves. The body's water and fat
contain hydrogen atoms, which respond to the magnetic field in measurable
ways. The scanner reads those responses and a computer turns them into
black-and-white images of soft tissue.

Because soft tissues — brain, muscle, ligaments, organs — show up clearly on
MRI, it is one of the best tools for looking at the brain.

## What you usually see in a brain MRI

- Slices through the brain from different angles (top-down, side, front).
- Different "weightings" (T1, T2, FLAIR, and others), each showing tissue in a
  different way. Radiologists look at multiple weightings together.
- Light and dark areas — what looks bright in one weighting may look dark in
  another, and vice versa.
- Sometimes a contrast agent (a special dye) is given to highlight certain
  tissues.

## Why brain MRIs are usually grayscale

Brain MRIs are produced in shades of gray — there is no real-world color
information in the signal. If a brain image is in color, that color was
typically added later for visualization or annotation. This chatbot's image
sanity check looks at color saturation as a quick filter against
non-MRI uploads.

## What this model does with an MRI

It compares the image to four kinds of examples it has seen during training
(glioma, meningioma, pituitary, no tumor) and outputs the closest match. It
does **not** read the scan the way a radiologist does. It does not see the
clinical context, the rest of the brain, or anything outside its narrow task.

This is not medical advice. Please consult a qualified clinician for any
medical decisions.
