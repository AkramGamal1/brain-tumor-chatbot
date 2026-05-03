# What this model cannot tell you

The model behind this chatbot is good at one narrow task: looking at a brain
MRI image and choosing among four labels (glioma, meningioma, pituitary, no
tumor). Almost nothing else you might want to know about a brain scan or a
person is something this model can answer.

## Things the model cannot do

- **Diagnose anyone.** Diagnosis is a clinical act involving a person's
  history, exam, labs, multiple images, and judgment. The model only sees
  pixels.
- **Rule out conditions outside its four categories.** It does not recognize
  strokes, bleeds, infections, MS, cysts, metastases, vascular malformations,
  trauma, infections, or many other tumor types. A "no tumor" result tells
  you nothing about any of those.
- **Determine grade, subtype, or aggressiveness** of any tumor it does flag.
  The label is just a category, not a description.
- **Predict outcomes.** It cannot say what is going to happen, what treatment
  will work, or how anyone will respond.
- **Recommend treatment.** The chatbot will refuse questions of the form
  "should I have surgery / chemo / radiation / this medication". Those
  decisions belong to clinical teams.
- **Comment on image quality** or whether an upload is "really" an MRI. There
  is a basic sanity check that flags obviously-non-grayscale images, but it is
  not a quality grader.
- **Replace a radiologist's read.** A radiologist looks at the whole brain in
  multiple sequences, with clinical context. The model looks at one image and
  picks one of four labels.

## Specific weaknesses to know about

- **Glioma vs. meningioma.** The two can look similar on MRI. The model has a
  documented tendency to confuse them when both are plausible. The chatbot
  surfaces this when it happens.
- **Overconfidence.** Some incorrect predictions come back with very high
  confidence numbers. That is why this chatbot doesn't show the raw numbers
  and uses verbal bands instead.
- **Out-of-distribution images.** Anything that isn't a brain MRI of the type
  the model was trained on (different scanner, different sequence, different
  view, different population) can produce unreliable results.

## What to do instead

For anything beyond "what does the model think this image looks most like" —
talk to a clinician. They can do what this model cannot.

This is not medical advice. Please consult a qualified clinician for any
medical decisions.
