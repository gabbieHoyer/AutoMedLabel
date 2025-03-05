#!/usr/bin/env python
import argparse

from src.finetuning.finetune_main import finetune_main
from src.finetuning.finetune_evaluate import finetune_evaluate
from src.finetuning.finetune_evaluate_biomarker import finetune_evaluate_biomarker
from src.finetuning.finetune_evaluate_det2seg import finetune_evaluate_det2seg

from src.obj_detection.train_obj_det import train_obj_det
from src.obj_detection.val_obj_det import validate_obj_det
from src.obj_detection.predict_obj_det import predict_obj_det
from src.obj_detection.autolabel import run_autolabel #autoLabel

# Note: We do NOT do GPU setup or config loading here.
# Those modules will handle their own setup when invoked.

def main():
    parser = argparse.ArgumentParser(description="Run a pipeline from the AutoMedLabel codebase")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline to run")

    # Finetuning training
    ft_parser = subparsers.add_parser("finetune", help="Run finetuning training")
    ft_parser.add_argument("config", help="YAML config file for finetuning experiment")

    # Finetuning evaluation
    eval_parser = subparsers.add_parser("eval", help="Run finetuning evaluation")
    eval_parser.add_argument("config", help="YAML config file for finetuning evaluation experiment")

    # Finetuning evaluation for biomarker metrics
    biom_parser = subparsers.add_parser("eval_biomarker", help="Run finetuning evaluation for biomarker metrics")
    biom_parser.add_argument("config", help="YAML config file for biomarker evaluation")

    # Finetuning evaluation using det2seg (detection-to-segmentation)
    det2seg_parser = subparsers.add_parser("eval_det2seg", help="Run finetuning evaluation using det2seg")
    det2seg_parser.add_argument("config", help="YAML config file for det2seg evaluation")

    # Object detection training
    train_det_parser = subparsers.add_parser("train_det", help="Train an object detection model")
    train_det_parser.add_argument("config", help="YAML config file for object detection training")

    # Object detection validation
    val_det_parser = subparsers.add_parser("val_det", help="Validate an object detection model")
    val_det_parser.add_argument("config", help="YAML config file for object detection validation")

    # Object detection prediction
    predict_det_parser = subparsers.add_parser("predict_det", help="Predict with an object detection model")
    predict_det_parser.add_argument("config", help="YAML config file for object detection prediction")

    # Autolabel pipeline
    autolabel_parser = subparsers.add_parser("autolabel", help="Run autolabel pipeline")
    autolabel_parser.add_argument("config", help="YAML config file for autolabel inference")
    autolabel_parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.command == "finetune":
        finetune_main(args.config)
    elif args.command == "eval":
        finetune_evaluate(args.config)
    elif args.command == "eval_biomarker":
        finetune_evaluate_biomarker(args.config)
    elif args.command == "eval_det2seg":
        finetune_evaluate_det2seg(args.config)
    elif args.command == "train_det":
        train_obj_det(args.config)
    elif args.command == "val_det":
        validate_obj_det(args.config)
    elif args.command == "predict_det":
        predict_obj_det(args.config)
    elif args.command == "autolabel":
        run_autolabel(args.config, interactive=args.interactive)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
