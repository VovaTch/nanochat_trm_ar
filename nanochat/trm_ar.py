# TODO: KV cache

from enum import Enum
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info
from nanochat.muon import DistMuon, Muon


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

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def init_weights(self) -> None:
        self.apply(self._init_weights)
        for block in self._transformer_encoder:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)  # type: ignore
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # type: ignore

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
        input_embedding: InputEmbedding,
        output_head: LinearOutputHead,
        q_head: LinearQOutputHead,
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

        self._loss_fn_cls = torch.nn.CrossEntropyLoss()
        self._loss_fn_q_stop = torch.nn.BCEWithLogitsLoss()

    def init_weights(self) -> None:
        """
        Initializes the model's weights
        """
        self._core.init_weights()
        torch.nn.init.zeros_(self._output_head._head.weight)
        if self._input_embedding._embedding.weight.device.type == "cuda":
            self._input_embedding.to(dtype=torch.bfloat16)

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

    def estimate_flops(self) -> int:
        """
        Estimates the number of flops activated per token

        Returns:
            int: Number of flops
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self._input_embedding._embedding.weight.numel()
        len, h, q, t = (
            self.core._num_layers,
            self.core._num_heads,
            self.core._hidden_dim // self.core._hidden_dim,
            self.core._seq_delimiter // 4,
        )
        num_flops_per_token = (
            6 * (nparams - nparams_embedding)
            + 12 * len * h * q * t * self._y_loop * self._z_loop
        )
        return num_flops_per_token

    def setup_optimizers(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Initializes the optimizers: AdamW and Muon, each for the corresponding parameters.

        Args:
            unembedding_lr (float, optional): Learning rate for the unembedding parameters. Defaults to 0.004.
            embedding_lr (float, optional): Learning rate for the embedding parameters. Defaults to 0.2.
            matrix_lr (float, optional): Learning rate for the matrix parameters. Defaults to 0.02.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.0.

        Returns:
            tuple: Tuple containing the AdamW optimizer and the Muon optimizer
        """
        model_dim = self.core._hidden_dim
        ddp, rank, _, _ = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.core._transformer_encoder.parameters())
        embedding_params = list(self._input_embedding.parameters())
        lm_head_params = list(self._output_head.parameters())
        q_output_head_params = list(self._q_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params) + len(q_output_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=q_output_head_params, lr=unembedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer: torch.optim.Optimizer = AdamWFactory(adam_groups, **adamw_kwargs)  # type: ignore
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer: torch.optim.Optimizer = MuonFactory(matrix_params, **muon_kwargs)  # type: ignore
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return tuple(optimizers)  # type: ignore

    def get_loss(
        self,
        output: torch.Tensor,
        q_stop: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Returns the auto-regressive loss for the TRM. Includes also QStop loss (might be interferring though)

        Args:
            output (torch.Tensor): Output tensor
            q_stop (torch.Tensor): QStop inference tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Auto-regressive loss
        """
        # loss_cls = self._loss_fn_cls(output, targets)
        loss_cls = F.cross_entropy(
            output.transpose(1, 2), targets, reduction=loss_reduction
        )

        pred_logits_prob = F.softmax(output, dim=-1)
        entropy = -torch.sum(
            pred_logits_prob * torch.log(pred_logits_prob + 1e-8), dim=-1
        )
        max_entropy = math.log(pred_logits_prob.size(-1))
        certainty = 1 - (entropy / max_entropy)
        target_halting = (torch.argmax(output, dim=-1) == targets) & (certainty >= 0.8)

        loss_q_stop = F.binary_cross_entropy_with_logits(
            q_stop.squeeze(), target_halting.float().squeeze(), reduction=loss_reduction
        )
        loss = loss_cls + loss_q_stop
        return loss

    def get_device(self) -> torch.device:
        """
        Returns the device of the model

        Returns:
            torch.device: Device of the model
        """
        return next(self.parameters()).device


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
    """
    Factory function to return the auto-regressive TRM model.

    Args:
        hidden_dim (int): The hidden dimension of the model.
        num_layers (int): The number of layers in the model.
        num_heads (int): The number of heads in the model.
        seq_delimiter (int): The sequence delimiter for the model, defaults to 4096.
        dropout (float): The dropout rate for the model, defaults to 0.1.
        vocab_size (int): The vocabulary size of the model, defaults to 65.
        y_loop (int): The number of deep recursion steps in the model, defaults to 6.
        z_loop (int): The number of shallow recursion steps in the model, defaults to 3.

    Returns:
        TinyRecursiveModel: The auto-regressive TRM model
    """
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
