# How to code a Transformer model for time series forecasting in PyTorch
## PyTorch implementation of Transformer model used in "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case"

This is the repo for the two Towards Data Science article called ["How to make a Transformer for time series forecasting with PyTorch"](https://kaspergroesludvigsen.medium.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e) and ["How to run inference with a PyTorch time series Transformer"]("https://medium.com/towards-data-science/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c")

The first article explains step by step how to code the Transformer model used in the paper "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case." The article uses the Transformer architecture diagram from the paper as the point of departure and shows step by step how to implement the model with PyTorch.

The second article explains how to use the time series Transformer at inference time where you don't know the decoder input values.

The sandbox.py file shows how to use the Transformer to make a training prediction on the data from the .csv file in "/data".

The inference.py file contains the function that takes care of inference, and the inference_example.py file shows a pseudo-ish code example of how to use the function during model validation and testing. 

