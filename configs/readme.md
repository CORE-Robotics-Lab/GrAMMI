input_obs: input observation type
seq_len: length of the lstm input
step_length: number of timesteps to skip between each head
num_heads: number of decoder heads
current_step: whether or not to return the current step in dataset output 
    if true: returns (input, current_red_loc, future_red_locations)
    if false: returns (input, future_red_locations)