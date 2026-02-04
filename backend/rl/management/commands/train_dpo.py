"""
Django Management Command: Train DPO Model

Usage:
    python manage.py train_dpo
    python manage.py train_dpo --epochs 5 --batch-size 2
    python manage.py train_dpo --config small_gpu --export
    python manage.py train_dpo --data-file path/to/exhaustive_data.json --ratings-file path/to/ratings.json
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Train DPO model on collected widget feedback"

    def add_arguments(self, parser):
        # Data source
        parser.add_argument(
            "--source",
            type=str,
            choices=["db", "file"],
            default="db",
            help="Data source: 'db' for database, 'file' for JSON files",
        )
        parser.add_argument(
            "--data-file",
            type=str,
            help="Path to exhaustive_data.json (required if source=file)",
        )
        parser.add_argument(
            "--ratings-file",
            type=str,
            help="Path to ratings.json (required if source=file)",
        )

        # Training config
        parser.add_argument(
            "--config",
            type=str,
            choices=["default", "small_gpu", "high_quality"],
            default="default",
            help="Training configuration preset",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of training epochs (overrides config)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Training batch size (overrides config)",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Learning rate (overrides config)",
        )
        parser.add_argument(
            "--min-samples",
            type=int,
            default=50,
            help="Minimum training samples required",
        )

        # Output
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Output directory for checkpoints",
        )
        parser.add_argument(
            "--pair-type",
            type=str,
            choices=["widget", "fixture", "both"],
            default="both",
            help="Type of training pairs to generate",
        )

        # Post-training
        parser.add_argument(
            "--export",
            action="store_true",
            help="Export to GGUF after training",
        )
        parser.add_argument(
            "--register-ollama",
            action="store_true",
            help="Register with Ollama after export (implies --export)",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default="cc-widget-selector",
            help="Model name for Ollama registration",
        )

        # Resume
        parser.add_argument(
            "--resume",
            type=str,
            help="Path to checkpoint to resume from",
        )

        # Misc
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Build dataset but don't train",
        )

    def handle(self, *args, **options):
        from rl.config import get_config
        from rl.dataset_builder import (
            prepare_training_dataset,
            get_dataset_stats,
        )
        from rl.trainer import CommandCenterDPOTrainer

        # Get configuration
        config = get_config(options["config"])

        # Apply overrides
        if options["epochs"]:
            config["dpo"]["num_epochs"] = options["epochs"]
        if options["batch_size"]:
            config["dpo"]["batch_size"] = options["batch_size"]
        if options["learning_rate"]:
            config["dpo"]["learning_rate"] = options["learning_rate"]

        self.stdout.write(f"Configuration: {options['config']}")
        self.stdout.write(f"  Epochs: {config['dpo']['num_epochs']}")
        self.stdout.write(f"  Batch size: {config['dpo']['batch_size']}")
        self.stdout.write(f"  Learning rate: {config['dpo']['learning_rate']}")
        self.stdout.write("")

        # Build dataset
        self.stdout.write("Building training dataset...")

        try:
            if options["source"] == "file":
                if not options["data_file"] or not options["ratings_file"]:
                    raise CommandError(
                        "--data-file and --ratings-file required when source=file"
                    )
                dataset = prepare_training_dataset(
                    source="file",
                    data_path=options["data_file"],
                    ratings_path=options["ratings_file"],
                    pair_type=options["pair_type"],
                    min_samples=options["min_samples"],
                )
            else:
                dataset = prepare_training_dataset(
                    source="db",
                    pair_type=options["pair_type"],
                    min_samples=options["min_samples"],
                )

        except ValueError as e:
            raise CommandError(str(e))
        except FileNotFoundError as e:
            raise CommandError(str(e))

        # Show stats
        train_size = len(dataset["train"]) if "train" in dataset else len(dataset)
        val_size = len(dataset["validation"]) if "validation" in dataset else 0

        self.stdout.write(self.style.SUCCESS(f"Dataset built successfully!"))
        self.stdout.write(f"  Training samples: {train_size}")
        self.stdout.write(f"  Validation samples: {val_size}")
        self.stdout.write("")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry run - skipping training"))
            return

        # Run training
        self.stdout.write("Starting DPO training...")
        self.stdout.write("This may take a while depending on your GPU...")
        self.stdout.write("")

        trainer = CommandCenterDPOTrainer(config=config)

        try:
            trainer.load_base_model()
        except ImportError as e:
            raise CommandError(
                f"Missing dependencies: {e}\n"
                "Install with: pip install torch transformers trl peft bitsandbytes"
            )

        train_data = dataset["train"] if "train" in dataset else dataset
        eval_data = dataset.get("validation") if hasattr(dataset, "get") else None

        result = trainer.train(
            train_dataset=train_data,
            eval_dataset=eval_data,
            output_dir=options["output_dir"],
            resume_from_checkpoint=options["resume"],
        )

        if not result.success:
            raise CommandError(f"Training failed: {result.error_message}")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Training complete!"))
        self.stdout.write(f"  Final loss: {result.final_loss:.4f}" if result.final_loss else "")
        self.stdout.write(f"  Checkpoint: {result.checkpoint_path}")
        self.stdout.write("")

        # Export if requested
        if options["export"] or options["register_ollama"]:
            self.stdout.write("Exporting to GGUF...")

            from rl.export import export_to_ollama

            export_result = export_to_ollama(
                checkpoint_path=result.checkpoint_path,
                model_name=options["model_name"],
                register=options["register_ollama"],
            )

            if export_result.success:
                self.stdout.write(self.style.SUCCESS("Export complete!"))
                self.stdout.write(f"  GGUF file: {export_result.gguf_path}")
                self.stdout.write(f"  Size: {export_result.file_size_mb:.1f} MB")
                if export_result.ollama_model_name:
                    self.stdout.write(f"  Ollama model: {export_result.ollama_model_name}")
                    self.stdout.write("")
                    self.stdout.write(
                        f"To use the new model, update OLLAMA_MODEL_FAST={options['model_name']} in .env"
                    )
            else:
                self.stdout.write(
                    self.style.WARNING(f"Export failed: {export_result.error_message}")
                )
