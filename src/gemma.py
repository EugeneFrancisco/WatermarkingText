"""
A concrete `LLM` implementation backed by Google's Gemma family via Hugging Face
`transformers`. The same class serves both the toy (local, CPU/MPS) and big
(Modal GPU) checkpoints -- swap them purely through the `configs` dict.

Config keys (all required -- the constructor indexes them directly and fails
loudly on a missing key):
    model_name (str): HF repo id, e.g. "google/gemma-3-270m" (toy) or
        "google/gemma-2-9b" (big).
    on_modal (bool): True when running on a fresh Modal machine. Changes where
        the HF access token is read from (see `_resolve_hf_token`).
    hf_token (str | None): Explicit HF access token for local runs, or None to
        fall back to the `HF_TOKEN` env var / cached `huggingface-cli login`.
        Ignored when on_modal is True (the token comes from the Modal secret).
    device (str): "cpu" | "mps" | "cuda".
    dtype (torch.dtype): Weights dtype, e.g. torch.bfloat16 on GPU, torch.float32
        on CPU.
    temperature (float): Sampling temperature used by `generate` (0 => greedy).

--------------------------------------------------------------------------------
Modal transition notes (gated weights)
--------------------------------------------------------------------------------
Gemma is gated: accept the license on Hugging Face once, then make a read-scope
token available wherever the code runs.

* Local:  the token is read from `configs["hf_token"]`, else the `HF_TOKEN`
          environment variable, else your cached `huggingface-cli login`
          credential (passing token=None lets `transformers` use the cache).
* Modal:  a fresh remote machine is NOT logged into your HF account, so a plain
          `from_pretrained` will 403. Store the token as a Modal secret and
          expose it as the `HF_TOKEN` env var, then set `on_modal=True`:

              modal secret create huggingface HF_TOKEN=hf_...

          In your Modal app, attach the secret to the function:

              @app.function(secrets=[modal.Secret.from_name("huggingface")])

          `_resolve_hf_token` then reads `os.environ["HF_TOKEN"]`.

To avoid re-downloading ~GBs of weights on every cold start, mount a Modal
Volume at the HF cache dir and set `HF_HOME` to it, e.g.:

    vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    @app.function(volumes={"/cache": vol}, secrets=[...])
    # and set env HF_HOME=/cache so from_pretrained caches there.
"""
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm import LLM


class GemmaModel(LLM):
    """Gemma-backed `LLM`, config-swappable between the toy and big checkpoints."""

    def __init__(self, configs: dict):
        super().__init__(configs)

        self.model_name: str = configs["model_name"]
        self.on_modal: bool = configs["on_modal"]
        self.device: str = configs["device"]
        self.dtype: torch.dtype = configs["dtype"]
        self.temperature: float = configs["temperature"]

        token = self._resolve_hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=token, dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

    # -- setup helpers --------------------------------------------------------

    def _resolve_hf_token(self) -> Optional[str]:
        """
        Find the Hugging Face access token for gated Gemma weights.

        On Modal, the token comes from a Modal secret exposed as the `HF_TOKEN`
        env var (a fresh machine has no cached login). Locally, we prefer an
        explicit config value, then `HF_TOKEN`, and finally fall back to the
        cached `huggingface-cli login` credential (signalled by returning None).
        """
        if self.on_modal:
            token = os.environ.get("HF_TOKEN")
            if token is None:
                raise RuntimeError(
                    "on_modal=True but HF_TOKEN is not set. Attach the Modal "
                    "secret: @app.function(secrets=[modal.Secret.from_name('huggingface')])"
                )
            return token
        return self.configs["hf_token"] or os.environ.get("HF_TOKEN")

    # -- LLM interface --------------------------------------------------------

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        Given a batched context window, returns the batched pre-softmax logits
        for the next tokens. This is where a watermark injects its bias/rule.

        Args:
            context: a B x sequence_length tensor of context token ids.

        Returns:
            A B x vocab_size tensor of pre-softmax logits.
        """
        context = context.to(self.device)
        logits = self.model(input_ids=context).logits  # B x seq_len x vocab
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, context: torch.Tensor, length: int) -> torch.Tensor:
        """
        Generate up to `length` new tokens for a single (un-batched) context,
        stopping early on the EOS token. Threads a KV cache through the loop so
        each step only forwards the newly generated token, keeping generation
        O(n) rather than O(n^2).

        Args:
            context: a sequence_length tensor of context token ids.
            length: the maximum number of new tokens to generate.

        Returns:
            A (sequence_length + n) tensor of token ids, n <= length.
        """
        # Add a batch dim of 1; the model always expects B x seq_len.
        tokens = context.to(self.device).unsqueeze(0)

        # Prime the cache on the full context, then feed one token at a time.
        out = self.model(input_ids=tokens, use_cache=True)
        past = out.past_key_values
        generated = []

        for _ in range(length):
            next_logits = out.logits[:, -1, :]
            next_token = self._sample_token(next_logits)  # 1 x 1
            generated.append(next_token)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            out = self.model(
                input_ids=next_token, past_key_values=past, use_cache=True
            )
            past = out.past_key_values

        full = torch.cat([tokens] + generated, dim=1)
        return full.squeeze(0).cpu()

    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """
        Decode a tensor of token ids back into readable text.

        Args:
            tokens: a sequence_length tensor of token ids.

        Returns:
            The decoded string (special tokens stripped).
        """
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Encode text into token ids under Gemma's tokenizer.

        Args:
            text: the string to encode.

        Returns:
            A sequence_length tensor of token ids (1D).
        """
        return self.tokenizer(text, return_tensors="pt").input_ids[0]

    # -- internals ------------------------------------------------------------

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Baseline next-token sampler for `generate`: temperature sampling, or
        greedy argmax when temperature is 0. Returns a 1 x 1 tensor.
        """
        if self.temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
