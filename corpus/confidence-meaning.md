# What "confidence" means here — and what it doesn't

When the model makes a prediction, it outputs a number that some interfaces
call a "confidence score". This chatbot **does not show that number to users**.
Instead, it translates the underlying value into a verbal description.

## The verbal bands

- **"fairly certain"** — the model's top probability is reasonably high
  compared to the other three classes. The model has seen many images during
  training that look like this one.
- **"moderately confident"** — the model leans toward one answer but the
  margin over the next-best answer is smaller. There is real ambiguity.
- **"uncertain"** — the model's top probability is not much higher than the
  next-best. The image could plausibly fit more than one class.

## Why we don't show the raw number

Three reasons:

1. **The model is documented to be overconfident on some wrong predictions.**
   Saying "98%" makes a wrong answer feel certain. Saying "fairly certain"
   keeps the door open to error.
2. **Probability is hard to interpret in the moment.** A non-clinician
   reading "85%" naturally reads it as "very likely correct." That isn't what
   it means in the context of a single neural-network output.
3. **Verbal bands are easier to act on.** They invite the right next step —
   talk to a doctor — instead of inviting you to compute your own odds.

## When the verbal band is suppressed entirely

If the model's top-2 predictions are **glioma** and **meningioma** and the gap
between them is small, the chatbot will not give a band at all. Instead it
will say there is real ambiguity between those two classes and that a
clinician needs to interpret the image. This is because of a known weakness
where the model has trouble telling glioma and meningioma apart.

## Confidence is not accuracy

A "fairly certain" prediction is not the same as a correct prediction. It is
the model's internal feeling about its own answer — and the model has been
wrong while feeling certain. Treat any prediction as a starting point for a
conversation with a clinician, not as a conclusion.

This is not medical advice. Please consult a qualified clinician for any
medical decisions.
