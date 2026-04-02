import click
import yaml

from fritz_ds_lib.cli.config import AppConfig
from fritz_ds_lib.core.names import DatasetType
from fritz_ds_lib.job import cv, evaluate, predict, train


@click.group()
@click.option(
    "--cfg",
    help="Path to the config file.",
    type=click.Path(exists=True),
    default=None,
)
@click.pass_context
def cli(ctx, cfg):
    """Entrypoint of the click application."""
    with open(cfg, "r") as fin:
        cfg = yaml.safe_load(fin)
    ctx.obj = AppConfig(**cfg)


@cli.command("train")
@click.option("--do-all/--no-all", default=False)
@click.pass_obj
def cli_train(cfg: AppConfig, do_all: bool) -> None:
    if do_all:
        train.train_all(cfg)
    else:
        train.train(cfg)


@cli.command("predict")
@click.option(
    "--dataset",
    type=click.Choice(["validation", "test"]),
    default="test",
)
@click.option("--do-all/--no-all", default=False)
@click.pass_obj
def cli_predict(cfg: AppConfig, dataset: DatasetType, do_all: bool) -> None:
    if do_all:
        predict.predict_all(cfg, dataset)
    else:
        predict.predict(cfg, dataset)


@cli.command("evaluate")
@click.option(
    "--dataset",
    type=click.Choice(["validation", "test"]),
    default="test",
)
@click.option("--do-all/--no-all", default=False)
@click.pass_obj
def cli_evaluate(cfg: AppConfig, dataset: DatasetType, do_all: bool) -> None:
    if do_all:
        evaluate.evaluate_all(cfg, dataset)
    else:
        evaluate.evaluate(cfg, dataset)


@cli.command("cv")
@click.option("--do-all/--no-all", default=False)
@click.pass_obj
def cli_cv(cfg: AppConfig, do_all: bool) -> None:
    if do_all:
        cv.cv_all(cfg)
    else:
        cv.cv(cfg)


if __name__ == "__main__":
    cli()
