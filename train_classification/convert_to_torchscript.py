from pathlib import Path

import torch
import typer


from model import ClassificationModel


def convert_to_torchscript(checkpoint_path, output_path):
    model = ClassificationModel()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )["state_dict"]

    model.eval()

    model.load_state_dict(checkpoint)
    model.half()

    with torch.jit.optimized_execution(True):
        model = torch.jit.script(model)
    model.save(output_path)


def main(
    # config_path: Path = typer.Argument("./config/baseline.yml", help="Path to config"),
    checkpoint_path: Path = typer.Argument(
        "experiments/angle/best.pt",
        help="Path to checkpoint",
    ),
    output_path: Path = typer.Argument(
        "angle_sota_model2.torchscript", help="Path to output file"
    ),
):
    try:
        convert_to_torchscript(checkpoint_path, output_path)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
