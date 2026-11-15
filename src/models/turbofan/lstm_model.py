# The LSTM network, the code that cuts engines into sequences, and the
# training loop. Kept in its own file so steps 12, 14, 16 and 17 all use
# exactly the same definitions.
#
# PyTorch models are normally written as a small class. This is the only class
# in the project; it just says "run the LSTM, take its output at the LAST
# cycle, and turn that into one number".
#
# The defaults (30 cycles, 64 units, 2 layers, dropout 0.2) are the settings
# step 12 used. Steps 16-17 try other values by passing them in.

import copy
import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 30          # the shortest test engine has 31 cycles, so 30 always fits
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
RUL_CAP = 125
MIN_TEST_LENGTH = 31  # shortest test engine; training windows start no later than this


class RULModel(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        if layers == 1:
            dropout = 0.0     # PyTorch only applies LSTM dropout BETWEEN layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x has shape (batch, sequence length, n_features)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]          # the network's summary after the last cycle
        return self.head(last_step).squeeze(1)


# ------------------------------------------------------------------ sequences

def one_sequence(values, end, seq_len):
    # Rows end-seq_len .. end-1 of one engine. If the engine does not have that
    # many cycles yet, the start is padded by repeating its FIRST reading
    # (i.e. "assume it looked like it did on day one").
    start = end - seq_len
    if start >= 0:
        return values[start:end]
    padding = np.repeat(values[:1], -start, axis=0)
    return np.concatenate([padding, values[:end]])


def make_train_sequences(data, feature_columns, seq_len=SEQ_LEN):
    # One sequence ending at every cycle of every engine, from cycle
    # min(seq_len, 31) onwards. The label is the RUL at the sequence's LAST
    # cycle, so the model only ever sees the past. For seq_len <= 31 no padding
    # is ever needed; for longer sequences the early ones are padded, so the
    # model has seen padded inputs before it meets a short test engine.
    first_end = min(seq_len, MIN_TEST_LENGTH)
    x_list = []
    y_list = []
    for engine in sorted(data["unit_number"].unique()):
        engine_rows = data[data["unit_number"] == engine].sort_values("cycle")
        values = engine_rows[feature_columns].values.astype(np.float32)
        labels = engine_rows["RUL"].values.astype(np.float32)
        for end in range(first_end, len(engine_rows) + 1):
            x_list.append(one_sequence(values, end, seq_len))
            y_list.append(labels[end - 1])
    return np.array(x_list), np.array(y_list)


def make_last_sequences(data, feature_columns, seq_len=SEQ_LEN):
    # Only the final seq_len cycles of each engine - used for the 100 test engines.
    x_list = []
    y_list = []
    engine_ids = []
    for engine in sorted(data["unit_number"].unique()):
        engine_rows = data[data["unit_number"] == engine].sort_values("cycle")
        values = engine_rows[feature_columns].values.astype(np.float32)
        labels = engine_rows["RUL"].values.astype(np.float32)
        x_list.append(one_sequence(values, len(engine_rows), seq_len))
        y_list.append(labels[-1])
        engine_ids.append(engine)
    return np.array(x_list), np.array(y_list), engine_ids


def make_cut_sequences(data, feature_columns, cut_table, seq_len=SEQ_LEN):
    # For cross-validation: the sequence ending at each cut cycle (see step 10),
    # i.e. exactly what the model would see if the engine had been cut off there.
    x_list = []
    y_list = []
    for engine in sorted(cut_table["unit_number"].unique()):
        engine_rows = data[data["unit_number"] == engine].sort_values("cycle")
        if len(engine_rows) == 0:
            continue
        values = engine_rows[feature_columns].values.astype(np.float32)
        labels = engine_rows["RUL"].values.astype(np.float32)
        cycles = list(engine_rows["cycle"].values)
        engine_cuts = cut_table[cut_table["unit_number"] == engine]["cycle"].values
        for cut in engine_cuts:
            end = cycles.index(cut) + 1
            x_list.append(one_sequence(values, end, seq_len))
            y_list.append(labels[end - 1])
    return np.array(x_list), np.array(y_list)


# ------------------------------------------------------------------ training

def train_one_epoch(model, optimizer, x_train, y_train, generator, batch_size):
    model.train()
    loss_function = nn.MSELoss()
    order = torch.randperm(len(x_train), generator=generator)
    for start in range(0, len(x_train), batch_size):
        batch_index = order[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_function(model(x_train[batch_index]), y_train[batch_index])
        loss.backward()
        optimizer.step()


def predict_cycles(model, x):
    # x can be a numpy array or a tensor; returns RUL in cycles, clipped to 0..125
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    model.eval()
    with torch.no_grad():
        predictions = model(x).numpy() * RUL_CAP
    return np.clip(predictions, 0, RUL_CAP)


def fit_early_stopping(x_fit, y_fit, x_stop, y_stop, n_features, seed,
                       hidden, layers, dropout, learning_rate, batch_size,
                       max_epochs=80, patience=10):
    # Train on (x_fit, y_fit); after every epoch measure RMSE on (x_stop, y_stop).
    # Stop when it has not improved for `patience` epochs and return the model
    # weights from the BEST epoch, plus that epoch number.
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x_fit = torch.tensor(x_fit)
    y_fit = torch.tensor(y_fit / RUL_CAP)
    x_stop = torch.tensor(x_stop)
    model = RULModel(n_features, hidden, layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_rmse = 1e9
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, max_epochs + 1):
        train_one_epoch(model, optimizer, x_fit, y_fit, generator, batch_size)
        rmse = np.sqrt(np.mean((predict_cycles(model, x_stop) - y_stop) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_weights)
    return model, best_epoch


def fit_fixed_epochs(x_all, y_all, n_features, seed, epochs,
                     hidden, layers, dropout, learning_rate, batch_size):
    # Train a fresh model for exactly `epochs` epochs (no validation set).
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    x_all = torch.tensor(x_all)
    y_all = torch.tensor(y_all / RUL_CAP)
    model = RULModel(n_features, hidden, layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, x_all, y_all, generator, batch_size)
    return model
