from collections import defaultdict

MODEL_PARAMS = defaultdict(
    lambda:None,
    d_model=512,
    n_layers=3,
    attention_dim=512,
    n_heads=8,
    hidden_size=2048,
    beam_size=5,
    length_penalty_weight=0.5,
    coverage_penalty_weight=0.5,
)

TRAIN_PARAMS = defaultdict(
    lambda:None,
    learning_rate = 0.0001,
    batch_size = 64,
    decay_rate = 0.5,
    decay_step = 5,
    dropout = None,
)

IKNOW_DATA_PARAMS = defaultdict(
    lambda : None,
    data_path = "iknow",
    validate_size = 1500,
    vocab_en_threshold = 2,
    vocab_ja_threshold = 2,
)

JESC_DATA_PARAMS = defaultdict(
    lambda : None,
    vocab_en_threshold = 10,
    vocab_ja_threshold = 10,
)

BASE_PARAMS = defaultdict(
    lambda:None,
    pad_id = 0,
    go_id = 1,
    eos_id = 2,
    unk_id = 3,
    tokenize_method = "",
    glove = False,
)
