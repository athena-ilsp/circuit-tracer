"""The Gemma-3-it chat prefix check accepts the image template's merged newline token."""

import pytest
import torch

from circuit_tracer.utils.tokenization import ensure_tokenized

# <bos> <start_of_turn> user, then "\n" (107) for text or "\n\n\n" (109) before an image.
_DECODED = {107: "\n", 109: "\n\n\n", 255999: "<start_of_image>", 818: "The"}


class _Tokenizer:
    all_special_ids = [2, 105]

    def decode(self, ids):
        return "".join(_DECODED.get(i, f"<{i}>") for i in ids)


@pytest.mark.parametrize("newline", [107, 109], ids=["text", "image-first"])
def test_gemma_3_it_accepts_a_newline_run_after_the_user_turn(newline):
    tokens = torch.tensor([2, 105, 2364, newline, 255999, 818])
    out = ensure_tokenized(tokens, _Tokenizer(), None, "google/gemma-3-4b-it")
    assert torch.equal(out, tokens)


def test_gemma_3_it_still_rejects_a_prompt_without_the_chat_prefix():
    with pytest.raises(AssertionError, match="should start with"):
        ensure_tokenized(
            torch.tensor([2, 105, 2364, 818]), _Tokenizer(), None, "google/gemma-3-4b-it"
        )
