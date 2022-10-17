"""
Pseudo code-ish example of how to use the inference function to do validation
during training. 

The validation loop can be used as-is for model testing as well.

NB! You cannot use this script as is. This is merely an example to show the overall idea - 
not something you can copy paste and expect to work. For instance, see "sandbox.py" 
for example of how to instantiate model and generate dataloaders.

If you have never before trained a PyTorch neural network, I suggest you look
at some of PyTorch's beginner-level tutorials.
"""
import torch
import inference
import utils

epochs = 10

forecast_window = 48 # supposing you're forecasting 48 hours ahead

enc_seq_len = 168 # supposing you want the model to base its forecasts on the previous 7 days of data

optimizer = torch.optim.Adam()

criterion = torch.nn.MSELoss()

# Iterate over all epochs
for epoch in range(epochs):

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_dataloader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
            )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
            )

        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()


    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():
    
        for i, (src, _, tgt_y) in enumerate(validation_dataloader):

            prediction = inference.run_encoder_decoder_inference(
                model=model, 
                src=src, 
                forecast_window=forecast_window,
                batch_size=src.shape[1]
                )

            loss = criterion(tgt_y, prediction)

