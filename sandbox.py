"""
Showing how to use the model with some time series data
"""

import data.dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np

# Hyperparams
test_size = 0.1
batch_size = 64 #128
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) # the data is consecutive all the way from datetime.datetime(2013, 1, 1), but HGT said to use from 2017 on

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 82#92 # length of input given to decoder
enc_seq_len = 123#153 # equivalent of what I called "input_len"
output_sequence_length = 48 #58
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len

# Define input variables 
#exogenous_vars = ["var2"] # user input from terminal - dummy var just for testing purposes
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# For testing multivariate input - Make dummy variable 
#training_data['var2'] = np.random.randint(0,5, size=len(training_data))

# Make list of (start_idx, end_idx) pairs. Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making dataloader
training_data = DataLoader(training_data, batch_size)

i, batch = next(enumerate(training_data))

src, trg, trg_y = batch

# In a multivariate setting before this slicing, trg_y contains all input variables. 
# It should only contain the target variable
# In a univariate setting it is redundant
# TODO: Consider controlling flow with if statement
#trg_y = trg_y[:, :, target_idx]


model = tst.TimeSeriesTransformer(
    dim_val=dim_val,
    input_size=input_size, 
    dec_seq_len=dec_seq_len,
    max_seq_len=max_seq_len,
    out_seq_len=output_sequence_length, 
    n_decoder_layers=n_decoder_layers,
    n_encoder_layers=n_encoder_layers,
    n_heads=n_heads)

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=batch_size*n_heads,
    dim2=output_sequence_length,
    dim3=enc_seq_len
    )

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=batch_size*n_heads,
    dim2=output_sequence_length,
    dim3=output_sequence_length
    )

output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )

assert output.size()[0] == batch_size and output.size()[1] == output_sequence_length

"""
### Training loop
epochs = 1


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(DEVICE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):

    for i, batch in enumerate(training_data):
        

        ## Train
        
        # Get model inputs
        src, trg, trg_y = batch

        src, trg, trg_y = src.to(DEVICE), trg.to(DEVICE), trg_y.to(DEVICE)

        output = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
            )

        loss = criterion(output, trg_y)

        if i % 5 == 0:
            print("LOSS --->: {}".format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



"""
