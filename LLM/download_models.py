#!/usr/bin/env python3
"""Download LLM models from HuggingFace to data/LLM/"""

from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Think",
]

def main():
    output_dir = Path("data/LLM")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        # Convert model name to safe directory name
        local_dir = output_dir / model_name.replace("/", "--")

        print(f"\n{'='*60}")
        print(f"Downloading: {model_name}")
        print(f"To: {local_dir}")
        print(f"{'='*60}")

        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Models saved to: {output_dir.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
