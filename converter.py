import torch
import whisper
from transformers import WhisperForConditionalGeneration
import argparse
import os

def hf_to_whisper_states(text: str) -> str:
    return (text
        .replace("model.", "")
        .replace("layers", "blocks")
        .replace("fc1", "mlp.0")
        .replace("fc2", "mlp.2")
        .replace("final_layer_norm", "mlp_ln")
        .replace(".self_attn.q_proj", ".attn.query")
        .replace(".self_attn.k_proj", ".attn.key")
        .replace(".self_attn.v_proj", ".attn.value")
        .replace(".self_attn_layer_norm", ".attn_ln")
        .replace(".self_attn.out_proj", ".attn.out")
        .replace(".encoder_attn.q_proj", ".cross_attn.query")
        .replace(".encoder_attn.k_proj", ".cross_attn.key")
        .replace(".encoder_attn.v_proj", ".cross_attn.value")
        .replace(".encoder_attn_layer_norm", ".cross_attn_ln")
        .replace(".encoder_attn.out_proj", ".cross_attn.out")
        .replace("decoder.layer_norm.", "decoder.ln.")
        .replace("encoder.layer_norm.", "encoder.ln_post.")
        .replace("embed_tokens", "token_embedding")
        .replace("encoder.embed_positions.weight", "encoder.positional_embedding")
        .replace("decoder.embed_positions.weight", "decoder.positional_embedding")
        .replace("layer_norm", "ln_post")
        .replace("proj_out.weight", "decoder.token_embedding.weight")
    )

def load_whisper_model(hf_model_dir: str, base_model_size: str = "small", use_gpu: bool = True, cache_dir: str = "./cache"):
    """
    Converts a fine-tuned HuggingFace Whisper model into OpenAI Whisper format
    and returns a usable whisper.Whisper instance.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Hugging Face model weights
    hf_model = WhisperForConditionalGeneration.from_pretrained(hf_model_dir)
    hf_state_dict = hf_model.state_dict()

    # Rename keys
    converted_state_dict = {}
    for key, value in hf_state_dict.items():
        new_key = hf_to_whisper_states(key)
        converted_state_dict[new_key] = value

    # Load base OpenAI Whisper architecture
    model = whisper.load_model(base_model_size, device=device, download_root=cache_dir)

    # Apply converted weights
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    print("Model converted successfully!")
    if missing:
        print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

    return model

def convert(hf_model_dir, base, out, cpu = False):
    model = load_whisper_model(
        hf_model_dir=hf_model_dir,
        base_model_size=base,
        use_gpu=not cpu,
        cache_dir="./cache"
    )

    # Save converted model
    torch.save({
        "dims": model.dims.__dict__,  # adds required metadata
        "model_state_dict": model.state_dict()
    }, out)
    print(f"Saved converted model as {out}")