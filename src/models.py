import torch
import torch.nn as nn


class CreditLSTMLogits(nn.Module):
   

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 4-layer LSTM with dropout between layers
        # Note: nn.LSTM dropout is applied between layers, not after the last layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only between layers
            batch_first=True
        )

        # 2 Dense layers: FC1 (hidden->64) + FC2 (64->1)
        # Paper: "two fully connected layers"
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

        # NO sigmoid here - output is logits
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights for better training convergence.
        
        LSTM weights:
            - Input-hidden: Xavier uniform
            - Hidden-hidden: Orthogonal (better for RNNs)
            - Biases: Zero
        
        Dense layers:
            - Weights: Xavier uniform
            - Biases: Zero
        """
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)  # Orthogonal is better for recurrent connections
            elif 'bias' in name:
                param.data.fill_(0)  # Zero biases
        
        # Initialize dense layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences (batch, seq_len, input_dim)
            lengths: Actual sequence lengths for packed sequence handling.
                     If None, uses the full sequence length.

        Returns:
            logits: Raw logits of shape (batch, 1). Apply sigmoid for probabilities.
        """
        batch_size = x.size(0)

        if lengths is not None:
            # Pack padded sequences for efficient LSTM processing
            # This correctly handles variable-length sequences
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)

            # Extract the last valid output for each sequence
            idx = (lengths - 1).to(x.device)
            last_out = lstm_out[torch.arange(batch_size, device=x.device), idx]
        else:
            # No lengths provided - use full sequence
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]  # Last timestep

        # Dense layers
        z = self.fc1(last_out)
        z = self.relu(z)
        z = self.dropout(z)
        logits = self.fc2(z)

        return logits


CreditLSTMPaper = CreditLSTMLogits
