"""Custom Vision-Language Model (VLM) main entry point."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom Vision-Language Model (VLM) constructed from ViT-B/16 and SmolLM2-360M-Instruct.",
    )
    parser.add_argument(
        "action",
        choices=[
            "check-gpu",
            "download-data",
            "prepare-data",
            "inspect",
            "test-forward",
            "test-gradients",
            "test-update",
            "test-real",
            "validate",
            "train",
            "generate",
            "ui",
        ],
        help="Action to execute.",
    )
    args, unknown = parser.parse_known_args()

    if args.action == "check-gpu":
        from scripts.check_gpu import check_gpu
        check_gpu()
    elif args.action == "download-data":
        from scripts.download_dataset import main as dl_main
        sys.argv = [sys.argv[0]] + unknown
        dl_main()
    elif args.action == "prepare-data":
        from scripts.prepare_dataset import main as prep_main
        sys.argv = [sys.argv[0]] + unknown
        prep_main()
    elif args.action == "inspect":
        from scripts.inspect_model import inspect_model
        inspect_model()
    elif args.action == "test-forward":
        from scripts.test_forward import test_forward
        test_forward()
    elif args.action == "test-gradients":
        from scripts.test_gradients import test_gradients
        test_gradients()
    elif args.action == "test-update":
        from scripts.test_parameter_update import test_parameter_update
        test_parameter_update()
    elif args.action == "test-real":
        from scripts.test_real_dataset import test_real_dataset_pipeline
        test_real_dataset_pipeline()
    elif args.action == "validate":
        from scripts.validate_model import main as val_main
        sys.argv = [sys.argv[0]] + unknown
        val_main()
    elif args.action == "train":
        from training.train import main as train_main
        sys.argv = [sys.argv[0]] + unknown
        train_main()
    elif args.action == "generate":
        from inference.generate import main as gen_main
        sys.argv = [sys.argv[0]] + unknown
        gen_main()
    elif args.action == "ui":
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])




if __name__ == "__main__":
    main()
