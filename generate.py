"""
Minimal image generation pipeline.
Usage:  python generate.py "a cyberpunk catgirl, neon lights, detailed"
"""

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Hugging Face model ID")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--output", type=str, default="outputs/output.png",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--negative", type=str, default="",
                        help="Negative prompt")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Memory optimizations for 16GB VRAM
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Generate
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"Generating: {args.prompt}")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative if args.negative else None,
        num_inference_steps=args.steps,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()