# model-hosting-challenge
This challenge requires the participant to host a image classification model in a python API framework of choice

## The Problem

In this repository is a trained sklearn SVC model that predicts which numeric symbol is handdrawn in an 8x8 pixel image.

The task is to host this model behind a API, which allows users to submit images (examples of which can be found in data) and have the image's class returned.

The image will have to be transformed for prediction as the model expects to take a (1, 64) vector of pixel intensities rather than RGB values. The input image of (8,8,3) will first need each pixel intensity calculated leaving an array of (8,8), that will then need to be unstacked into (1, 64). Values in the model input array should be integers that range from 0 to 16, white to black.

The model has been pickled and can be reloaded by the following

```python
import pickle
with open("model/classifier.pickle", "rb") as handle:
    classifier = pickle.load(handle)
```

Along with the API, tests should be written that show the API works as expected, and these tests can be written in a framework's testing suite or via unittest/pytest.

## Extras

To take this a little further, you could

- Handle input images of varying size
- Define a dockerfile for your application
