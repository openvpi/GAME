import pathlib

import click


@click.command(help="Deploy model for production use.")
@click.option(
    "-m", "--model", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the model."
)
@click.option(
    "-o", "--save-dir", type=click.Path(
        exists=False, dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Directory to save evaluation results."
)
@click.option(
    "--opset-version", type=click.IntRange(min=17),
    default=24, show_default=True,
    help="ONNX opset version to use for export."
)
def main(
        model: pathlib.Path,
        save_dir: pathlib.Path,
        opset_version: int,
):
    from inference.api import load_inference_model
    from deployment.api import deploy_model

    model, lang_map = load_inference_model(model)
    deploy_model(
        model=model, lang_map=lang_map, save_dir=save_dir,
        opset_version=opset_version,
    )


if __name__ == '__main__':
    main()
