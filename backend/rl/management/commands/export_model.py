"""
Django Management Command: Export Model to GGUF

Usage:
    python manage.py export_model --checkpoint ./rl_checkpoints/final
    python manage.py export_model --checkpoint ./rl_checkpoints/final --quantization q5_k_m
    python manage.py export_model --checkpoint ./rl_checkpoints/final --register
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Export trained model to GGUF format for Ollama deployment"

    def add_arguments(self, parser):
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to LoRA checkpoint directory",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default="cc-widget-selector",
            help="Name for the exported model",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default="q4_k_m",
            choices=["q2_k", "q3_k_s", "q3_k_m", "q4_0", "q4_k_s", "q4_k_m", "q5_0", "q5_k_s", "q5_k_m", "q6_k", "q8_0", "f16"],
            help="GGUF quantization level",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Output directory for GGUF file",
        )
        parser.add_argument(
            "--base-model",
            type=str,
            help="Base model path (defaults to config)",
        )
        parser.add_argument(
            "--register",
            action="store_true",
            help="Register model with Ollama after export",
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List available Ollama models and exit",
        )

    def handle(self, *args, **options):
        from pathlib import Path

        from rl.export import (
            export_to_ollama,
            list_ollama_models,
            get_export_status,
        )

        # List models mode
        if options["list_models"]:
            status = get_export_status()

            self.stdout.write("GGUF Files:")
            if status["gguf_files"]:
                for f in status["gguf_files"]:
                    self.stdout.write(f"  {f['name']} ({f['size_mb']:.1f} MB)")
            else:
                self.stdout.write("  (none)")

            self.stdout.write("")
            self.stdout.write("Ollama Models:")
            if status["ollama_models"]:
                for m in status["ollama_models"]:
                    self.stdout.write(f"  {m}")
            else:
                self.stdout.write("  (none)")

            return

        # Validate checkpoint
        checkpoint = Path(options["checkpoint"])
        if not checkpoint.exists():
            raise CommandError(f"Checkpoint not found: {checkpoint}")

        self.stdout.write(f"Exporting checkpoint: {checkpoint}")
        self.stdout.write(f"Model name: {options['model_name']}")
        self.stdout.write(f"Quantization: {options['quantization']}")
        self.stdout.write("")

        # Run export
        result = export_to_ollama(
            checkpoint_path=str(checkpoint),
            model_name=options["model_name"],
            base_model=options["base_model"],
            quantization=options["quantization"],
            output_dir=options["output_dir"],
            register=options["register"],
        )

        if result.success:
            self.stdout.write(self.style.SUCCESS("Export complete!"))
            self.stdout.write(f"  GGUF file: {result.gguf_path}")
            self.stdout.write(f"  Size: {result.file_size_mb:.1f} MB")

            if result.ollama_model_name:
                self.stdout.write(f"  Registered as: {result.ollama_model_name}")
                self.stdout.write("")
                self.stdout.write("To use the model:")
                self.stdout.write(f"  1. Update .env: OLLAMA_MODEL_FAST={result.ollama_model_name}")
                self.stdout.write("  2. Restart the backend")
                self.stdout.write("")
                self.stdout.write(f"Or test with: ollama run {result.ollama_model_name}")
            else:
                self.stdout.write("")
                self.stdout.write("To register with Ollama:")
                self.stdout.write(
                    f"  python manage.py export_model --checkpoint {checkpoint} --register"
                )
        else:
            raise CommandError(f"Export failed: {result.error_message}")
