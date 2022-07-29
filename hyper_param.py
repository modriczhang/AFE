#!encoding:utf-8

"""
    Model Hyper Parameter Dict
    @2020-02-24
    by Modric Zhang
"""
param_dict = {
    'booster_feat_dim': 16,  # feature embedding dimension
    'lite_feat_dim': 8,  # feature embedding dimension
    'user_field_num': 5,  # number of user feature fields
    'doc_field_num': 5,  # number of doc feature fields
    'con_field_num': 5,  # number of context feature fields
    'future_field_num': 2,  # number of future feature fields
    'num_epochs': 1,  # training epoch
    'batch_size': 256,  # batch size
    'booster_lr': 0.0002,  # learning rate
    'lite_lr': 0.0002,  # learning rate
    'hint_dim': 32,  # learning rate
    'dropout': 0.3,  # learning rate of q/value network
    'rl_gamma': 0.6,  # gamma in rl
    'grad_clip': 5,  # grad clip
    'booster_rnn_dim': 160,  # rnn dimension of booster network
    'lite_rnn_dim': 120,  # rnn dimension of light network
    'clk_seq_len': 10,  # click sequence length
    'rnn_seq_len': 10,  # rnn sequence length
    'head_num': 4,  # head number for all self-attention units
    'rnn_layer': 1,  # RNN layer
    'encoder_layer': 1,  # encoder layer
    'decoder_layer': 1,  # decoder layer
    'double_networks_sync_freq': 10,  # sync frequency for both policy and value network
    'double_networks_sync_step': 0.005,  # sync step is designed with reference to deepmind
    'enable_gan': True,
    'enable_distill': True,
    'enable_future': True,
}
