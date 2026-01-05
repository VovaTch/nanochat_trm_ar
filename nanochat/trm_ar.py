# TODO: KV cache

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingType(Enum):
    ROPE = "rope"
    SIN = "sin"


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """
    RMS Norm implementation from the TRM repo.

    Args:
        hidden_states (torch.Tensor): Input tensor
        variance_epsilon (float): Epsilon value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()

        if dim % 2:
            raise ValueError(f"Dimension must be divisible by 2, got {dim}")

        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).type_as(inv_freq)
        frequencies = torch.einsum("i,j->ij", t, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)

        # Get cached sin and cos values
        self.register_buffer("cached_cos", embeddings.cos(), persistent=False)
        self.register_buffer("cached_sin", embeddings.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int | None = None) -> torch.Tensor:
        """
        Forward method for RoPE embeddings; offset is unused
        """

        seq_len = x.shape[1]  # Assuming BxSxD dims

        if x.dim() == 3:
            cos = self.cached_cos[:seq_len, :]  # type: ignore
            sin = self.cached_sin[:seq_len, :]  # type: ignore
        elif x.dim() == 4:
            # Handle batch_size x num_heads x seq_len x dim case
            cos = self.cached_cos[None, :seq_len, None, :]  # type: ignore
            sin = self.cached_sin[None, :seq_len, None, :]  # type: ignore
        else:
            raise ValueError(f"Unsupported input with {x.dim()} dimensions")

        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((x1, x2), dim=-1)


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Pure RMS norm without learned parameters.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Normalized tensor
    """
    return F.rms_norm(x, (x.size(-1),))


class FullSelfAttention(nn.Module):
    """
    Full attention implementation from Andrej Karpathy's Nanochat
    """

    def __init__(
        self,
        n_head: int,
        n_kv_head: int,
        n_embd: int,
        layer_idx: int,
        max_seq_len: int = 2048,
        is_causal: bool = True,
    ) -> None:
        """
        Initializes the attention module

        Args:
            n_head (int): Number of attention heads
            n_kv_head (int): Number of key/value heads
            n_embd (int): Embedding dimension
            layer_idx (int): Layer index
            max_seq_len (int, optional): Maximum sequence length. Defaults to 2048.
            is_causal (bool, optional): Whether to use causal attention. Defaults to True.
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        self._max_seq_len = max_seq_len
        self._is_causal = is_causal

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self._rot_embedding_q = RotaryEmbedding(self.head_dim, max_seq_len)
        self._rot_embedding_k = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = self._rot_embedding_q(q), self._rot_embedding_k(k)  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self._is_causal)
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron implementation from Karpathy's Nanochat
    """

    def __init__(self, n_embd: int, mlp_multiplier: int = 4) -> None:
        """
        Initializes the MLP class

        Args:
            n_embd (int): Embedding dimension
            mlp_multiplier (int, optional): Multiplier for the number of units in the MLP. Defaults to 4.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, mlp_multiplier * n_embd, bias=False)
        self.c_proj = nn.Linear(mlp_multiplier * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer block implementation from Karpathy's Nanochat
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        layer_idx: int,
        max_seq_len: int = 2048,
        mlp_multiplier: int = 4,
        is_causal: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the Transformer Block class
        """
        super().__init__()
        self.attn = FullSelfAttention(
            num_heads, num_kv_heads, hidden_dim, layer_idx, max_seq_len, is_causal
        )
        self.mlp = MLP(hidden_dim, mlp_multiplier)
        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(self._dropout(x)))
        x = x + self.mlp(norm(self._dropout(x)))
        return x


class ARTransformerTRM(nn.Module):
    """
    Diffusion Transformer TRM core for using it for the TRM model
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        seq_delimiter: int = 4096,
        dropout: float = 0.1,
        vocab_size: int = 65,
    ) -> None:
        """
        Initializer

        Args:
            hidden_dim (int): The hidden dimension of the model.
            vocab_size (int): The vocabulary size of the model.
            latent_len (int): The latent length of the model.
            num_layers (int): The number of layers in the model.
            num_heads (int): The number of heads in the model.
            seq_delimiter (int): The sequence delimiter for the model, defaults to 4096.
                Used for positional embedding of the output and the latent. Sequence length
                should not exceed it.
            dropout (float): The dropout rate for the model, defaults to 0.1.
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._seq_delimiter = seq_delimiter
        self._vocab_size = vocab_size

        self._transformer_encoder = nn.Sequential(
            *[
                Block(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    layer_idx=i,
                    is_causal=True,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        self._y_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)
        self._z_init = nn.Buffer(torch.randn((1, 1, hidden_dim)), persistent=True)

    @property
    def y_init(self) -> nn.Buffer:
        return self._y_init

    @property
    def z_init(self) -> nn.Buffer:
        return self._z_init

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x is not None:

            input_sum = x + y + z

        else:

            input_sum = y + z

        transformer_output = self._transformer_encoder(input_sum)
        output = rms_norm(transformer_output, 1e-6)

        return output, output

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class InputEmbedding(nn.Module):
    def __init__(
        self, embedding_dim: int, vocab_size: int, additional_emb: bool = True
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        add_emb = 1 if additional_emb else 0
        self._embedding = nn.Embedding(vocab_size + add_emb, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x)


class LinearOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)


class LinearQOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, seq_length: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_length = seq_length

        layers = []
        layers.append(nn.Linear(hidden_dim, 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers[0](x)
        return x.view(x.shape[0], -1)


class TinyRecursiveModel(nn.Module):
    """
    General TRM implementation, with support for latent recursion, deep recursion, and a general forward function
    """

    def __init__(
        self,
        core: ARTransformerTRM,
        z_loop: int,
        y_loop: int,
        input_embedding: nn.Module,
        output_head: nn.Module,
        q_head: nn.Module,
    ) -> None:
        """
        Initializer

        Args:
            core (Core): core module
            z_loop (int): number of latent reasoning steps
            y_loop (int): number of deep recursion steps
            input_embedding (nn.Module): input embedding module
            output_head (nn.Module): output head from the network output
            q_head (nn.Module): network cut-off head
        """
        super().__init__()
        self._core = core
        self._z_loop = z_loop
        self._y_loop = y_loop
        self._input_embedding = input_embedding
        self._output_head = output_head
        self._q_head = q_head

    @property
    def core(self) -> ARTransformerTRM:
        """
        Returns the core underlying model of the TRM.
        """
        return self._core

    def latent_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Latent recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        for _ in range(self._z_loop):
            _, latent = self._core(input, output, latent)
        output, _ = self._core(None, output, latent)
        return output, latent

    def deep_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion method of the TRM.

        Args:
            input (torch.Tensor): Input tensor
            output (torch.Tensor): Output tensor
            latent (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor, latent tensor, output head tensor, and stop tensor
        """
        input = self._input_embedding(input)
        with torch.no_grad():
            for _ in range(self._y_loop - 1):
                output, latent = self.latent_recursion(input, output, latent)
        output, latent = self.latent_recursion(input, output, latent)
        return (
            output.detach(),
            latent.detach(),
            self._output_head(output),
            self._q_head(output),
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method of the TRM.

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Output tensor
            z (torch.Tensor): Latent tensor

        Returns:
            tuple: Tuple containing the output tensor and latent tensor
        """
        _, _, output, latent = self.deep_recursion(x, y, z)
        return output, latent


def get_trm_ar_model(
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    seq_delimiter: int = 4096,
    dropout: float = 0.1,
    vocab_size: int = 65,
    y_loop: int = 6,
    z_loop: int = 3,
) -> TinyRecursiveModel:
    core = ARTransformerTRM(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        vocab_size=vocab_size,
        seq_delimiter=seq_delimiter,
    )
    input_embedding = InputEmbedding(embedding_dim=hidden_dim, vocab_size=vocab_size)
    output_head = LinearOutputHead(hidden_dim=hidden_dim, vocab_size=vocab_size)
    q_head = LinearQOutputHead(hidden_dim=hidden_dim, seq_length=seq_delimiter)
    return TinyRecursiveModel(
        core=core,
        z_loop=z_loop,
        y_loop=y_loop,
        input_embedding=input_embedding,
        output_head=output_head,
        q_head=q_head,
    )
