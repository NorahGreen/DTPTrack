def mamba_flop_jit(inputs, outputs):
    B, _, seq_len = inputs[0].type().sizes()
    d_state = inputs[6].type().sizes()[1]
    d_model = outputs[0].type().sizes()[2]
    return B * 9 * d_state * d_model * seq_len