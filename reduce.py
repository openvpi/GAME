import pathlib

import click


@click.command(help="Reduce a checkpoint file, only keeping the 'state_dict' key for inference.")
@click.argument(
    "input_ckpt", metavar="INPUT_PATH", type=click.Path(
        exists=True, dir_okay=False, readable=True, path_type=pathlib.Path
    ),
)
@click.argument(
    "output_ckpt", metavar="OUTPUT_PATH",
    type=click.Path(
        exists=False, dir_okay=False, writable=True, path_type=pathlib.Path
    ),
)
def reduce(input_ckpt, output_ckpt):
    import torch
    input_ckpt_path = pathlib.Path(input_ckpt)
    output_ckpt_path = pathlib.Path(output_ckpt)
    ckpt = torch.load(input_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict")
    if (ema_state_dict := ckpt.get("ema_state_dict")) is not None:
        state_dict = ema_state_dict
    if state_dict is None:
        raise ValueError(f"Checkpoint file '{input_ckpt_path}' does not contain 'state_dict' or 'ema_state_dict'.")
    ckpt = {
        "state_dict": ckpt["state_dict"]
    }
    torch.save(ckpt, output_ckpt_path)


if __name__ == "__main__":
    reduce()
