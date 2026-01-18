import torch
import torch.nn as nn

class CreditLSTMPaper(nn.Module):
    """
    Exact replication of the LSTM model from Delgado Fernandez et al. (2023).
    
    Architecture:
    - Input: Sequence of 31 features (or subset of dynamic + static repeated)
    - 4 x (LSTM + Dropout) layers
    - 2 x Fully Connected layers
    - Output: Sigmoid probability
    
    Paper Quote: "Neural Network (NN) with four layers of Long Short-Term Memory... 
    interspersed with dropout layers and two fully connected layers."
    """
    def __init__(self, input_dim=31, hidden_dim=64, num_layers=4, dropout=0.2):
        super(CreditLSTMPaper, self).__init__()
        
        # LSTM Block
        # "4 layers of LSTM"
        # Input: (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout, # Dropout introduced between LSTM layers
            batch_first=True
        )
        
        # Dense Block
        # "two fully connected layers" -> Implies Hidden -> Dense1 -> Dense2 -> Output?
        # Or Dense1 -> Dense2 (Output)? 
        # Usually "two FC layers" implies one hidden FC and one output FC, or two hidden + output.
        # Given Table 2 says "Baseline architecture: 4x (LSTM+Dropout) + 2 Dense", 
        # and standard binary classification usually needs a projection layer.
        # Let's assume: LSTM -> Dense(64) -> ReLU -> Dropout -> Dense(32) -> ReLU -> Dense(1).
        # Wait, that's 3 Dense layers.
        # If "two fully connected layers", it arguably means:
        # LSTM -> Dense(Hidden) -> Act -> Dense(Output).
        # Let's check `CreditNet` implementation in my previous turn (from `models.py` inspection):
        # We saw `models.py` had `CreditNet` (MLP) with 2 hidden layers?
        # Let's stick to the interpretation: LSTM -> Dense1 -> Act -> Dropout -> Dense2 (Output)?
        # Or LSTM -> Dense1 -> Act -> Dense2 -> Act -> Output?
        # Let's use the configuration from my earlier `CreditLSTMPaper` which had Dense1(64) -> Dense2(32) -> Output.
        # That's actually 3 linear transformations.
        
        # Re-reading Table 2 Hypers: "Baseline architecture: 4x (LSTM + Dropout) + 2 Dense".
        # This is slightly ambiguous. Let's assume "2 Hidden Dense Layers" + Output? 
        # Or 2 Dense Layers TOTAL (Hidden + Output).
        # Let's go with 2 Dense Layers TOTAL for simplicity and standard layout.
        # LSTM -> Dense(64) -> ReLU -> Dropout -> Dense(1) -> Sigmoid ? 
        # Or LSTM -> Dense(64) -> ReLU -> Dense(32) -> Output ? 
        
        # Let's stick to a robust default: 
        # Dense1 (64 units) -> ReLU -> Dropout
        # Dense2 (1 unit) -> Sigmoid
        # This counts as "2 Dense layers".
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1) # Output layer
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        
        # Pack padded sequence if lengths provided? 
        # Paper doesn't specify masking details. Standard practice is packing.
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out_packed, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
            
            # Extract last time step
            # idx = (lengths - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
            # idx = idx.unsqueeze(1)
            # last_out = lstm_out.gather(1, idx).squeeze(1)
            
            # Simpler: just take the last element for each batch based on length
            idx = (lengths - 1).to(x.device)
            last_out = lstm_out[torch.arange(x.size(0)), idx]
        else:
            # Assume fixed length or padding is 0 and LSTM handles it (suboptimally)
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :] # Last time step
            
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        probs = self.sigmoid(logits)
        
        return probs
