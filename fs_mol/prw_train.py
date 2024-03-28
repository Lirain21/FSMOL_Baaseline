import argparse
import logging
import sys
import json
from pathlib import Path

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
PAR_CODE_PATH = "./PAR-NeurIPS21"
assert Path(PAR_CODE_PATH).exists()
sys.path.insert(0, PAR_CODE_PATH)


from fs_mol.modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
from fs_mol.utils.prw_utils import (
    PRWModelTrainerConfig,
    PRWModelTrainer,
)


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train a PAR model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)

    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "gnn",
            "ecfp",
            "pc-descs",
            "ecfp+fc",
            "pc-descs+fc",
            "gnn+ecfp+fc",
            "gnn+ecfp+pc-descs+fc",
        ],
        default="gnn+ecfp+fc",
        help="Choice of features to use",
    )
    add_graph_feature_extractor_arguments(parser)

    parser.add_argument("--support_set_size", type=int, default=16, help="Size of support set")   
    parser.add_argument(
        "--query_set_size",
        type=int,
        default=256,
        help="Size of target set. If -1, use everything but train examples.",
    )
    parser.add_argument(
        "--unlabeled_set_size",
        type=int,
        default=10,
        help="Size of unlabeled set."
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=9,  # from their code
        help="Number of tasks to accumulate gradients for.",
    )

    parser.add_argument("--batch_size", type=int, default=512, help="Number of examples per batch.")
    parser.add_argument(
        "--num_train_steps", type=int, default=10000, help="Number of training steps."
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=50,
        help="Number of training steps between model validations.",
    )
    parser.add_argument(
        "--validation-support-set-sizes",
        type=json.loads,
        #default=[16, 128],
        default=[16, 64],
        help="JSON list selecting the number of datapoints sampled as support set data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-query-set-size",
        type=int,
        #default=512,
        default=256,
        help="Maximum number of datapoints sampled as query data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-num-samples",
        type=int,
        default=5,
        help="Number of samples considered for each train set size for each validation task during evaluation through finetuning.",
    )
    parser.add_argument("--lr_outer", type=float, default=1e-3, help="Outer Learning rate")
    parser.add_argument("--lr_inner", type=float, default=1e-1, help="Inner MAML Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    parser.add_argument(
        "--pretrained_gnn",
        type=str,
        default=None,
        help="Path to a pretrained GNN model to use as a starting point.",
    )
    parser.add_argument(
        "--use-numeric-labels",
        action="store_true",
        help="Perform regression for the numeric labels (log concentration). Default: perform binary classification for the bool labels (active/inactive).",
    )
    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> PRWModelTrainerConfig:
    if args.use_numeric_labels:
        raise NotImplementedError
    return PRWModelTrainerConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        batch_size=args.batch_size,
        tasks_per_batch=args.tasks_per_batch,
        support_set_size=args.support_set_size,
        query_set_size=args.query_set_size,
        unlabeled_set_size = args.unlabeled_set_size,
        validate_every_num_steps=args.validate_every,
        validation_support_set_sizes=tuple(args.validation_support_set_sizes),
        validation_query_set_size=args.validation_query_set_size,
        validation_num_samples=args.validation_num_samples,
        num_train_steps=args.num_train_steps,
        outer_learning_rate=args.lr_outer,
        inner_learning_rate=args.lr_inner,
        clip_value=args.clip_value,
        use_numeric_labels=args.use_numeric_labels,
    )


def main():
    args = parse_command_line()
    config = make_trainer_config(args)

    out_dir, dataset, aml_run = set_up_train_run(
        f"PRWModel_{config.used_features}", args, torch=True
    )

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model_trainer = PRWModelTrainer(config=config).to(device)

    logger.info(f"\tDevice: {device}")
    logger.info(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    logger.info(f"\tModel:\n{model_trainer}")

    if args.pretrained_gnn is not None:
        logger.info(f"Loading pretrained GNN weights from {args.pretrained_gnn}.")
        model_trainer.load_model_gnn_weights(path=args.pretrained_gnn, device=device)

    model_trainer.train_loop(out_dir, dataset, device, aml_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)