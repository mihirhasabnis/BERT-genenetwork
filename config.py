maxlen = 15 # maximum length
batch_size = 6
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 2 # number of heads in Multi-Head Attention
d_model = 32 # Embedding Size
vocab_size = 21552
d_ff = 32 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 16  # dimension of K(=Q), V, tpyically d_model//n_heads
n_segments = 2
epochs = 2
model_path = "model.bin"
training_data = r"C:\Downloads\LLM_Sridhar\BERT-genenetwork\rand_walks_samplewise\rand_walk_1.txt"
