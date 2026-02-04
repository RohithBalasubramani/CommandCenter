"""
Django Management Command: Evaluate Model

Usage:
    python manage.py evaluate_model --checkpoint ./rl_checkpoints/final
    python manage.py evaluate_model --checkpoint ./rl_checkpoints/final --test-file test_data.jsonl
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Evaluate trained model on test data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="Path to LoRA checkpoint (uses base model if not specified)",
        )
        parser.add_argument(
            "--test-file",
            type=str,
            help="Path to test data JSONL file",
        )
        parser.add_argument(
            "--data-file",
            type=str,
            help="Path to exhaustive_data.json for generating test set",
        )
        parser.add_argument(
            "--ratings-file",
            type=str,
            help="Path to ratings.json for generating test set",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=100,
            help="Number of samples to evaluate",
        )
        parser.add_argument(
            "--pair-type",
            type=str,
            choices=["widget", "fixture", "both"],
            default="both",
            help="Type of pairs to evaluate",
        )
        parser.add_argument(
            "--compare-base",
            action="store_true",
            help="Also evaluate base model for comparison",
        )
        parser.add_argument(
            "--output",
            type=str,
            help="Output file for detailed results (JSON)",
        )

    def handle(self, *args, **options):
        import json
        from pathlib import Path

        from rl.config import get_config
        from rl.dataset_builder import prepare_training_dataset
        from rl.data_formatter import load_pairs_from_jsonl

        config = get_config()

        # Load or generate test data
        self.stdout.write("Loading test data...")

        test_pairs = None
        if options["test_file"]:
            test_file = Path(options["test_file"])
            if not test_file.exists():
                raise CommandError(f"Test file not found: {test_file}")
            test_pairs = load_pairs_from_jsonl(str(test_file))
        elif options["data_file"] and options["ratings_file"]:
            dataset = prepare_training_dataset(
                source="file",
                data_path=options["data_file"],
                ratings_path=options["ratings_file"],
                pair_type=options["pair_type"],
                min_samples=10,
                val_split=1.0,  # Use all as test
            )
            # Convert to pairs
            from rl.data_formatter import DPOPair
            test_pairs = [
                DPOPair(
                    prompt=row["prompt"],
                    chosen=row["chosen"],
                    rejected=row["rejected"],
                    question_id=row["question_id"],
                )
                for row in (dataset["validation"] if "validation" in dataset else dataset)
            ]
        else:
            # Try database
            try:
                dataset = prepare_training_dataset(
                    source="db",
                    pair_type=options["pair_type"],
                    min_samples=10,
                    val_split=0.3,
                )
                from rl.data_formatter import DPOPair
                test_pairs = [
                    DPOPair(
                        prompt=row["prompt"],
                        chosen=row["chosen"],
                        rejected=row["rejected"],
                        question_id=row["question_id"],
                    )
                    for row in dataset.get("validation", [])
                ]
            except Exception as e:
                raise CommandError(f"Could not load test data: {e}")

        if not test_pairs:
            raise CommandError("No test data available")

        # Limit samples
        if len(test_pairs) > options["num_samples"]:
            import random
            random.seed(42)
            test_pairs = random.sample(test_pairs, options["num_samples"])

        self.stdout.write(f"Evaluating on {len(test_pairs)} samples...")
        self.stdout.write("")

        # Evaluate
        results = {
            "num_samples": len(test_pairs),
            "checkpoint": options["checkpoint"],
            "metrics": {},
        }

        if options["checkpoint"]:
            self.stdout.write("Loading fine-tuned model...")
            from rl.trainer import CommandCenterDPOTrainer

            trainer = CommandCenterDPOTrainer()
            trainer.load_checkpoint(options["checkpoint"])

            # Run evaluation
            finetuned_metrics = self._evaluate_model(trainer, test_pairs)
            results["metrics"]["finetuned"] = finetuned_metrics

            self.stdout.write(self.style.SUCCESS("Fine-tuned model results:"))
            self._print_metrics(finetuned_metrics)

        if options["compare_base"] or not options["checkpoint"]:
            self.stdout.write("")
            self.stdout.write("Loading base model...")
            from rl.trainer import CommandCenterDPOTrainer

            base_trainer = CommandCenterDPOTrainer()
            base_trainer.load_base_model()

            base_metrics = self._evaluate_model(base_trainer, test_pairs)
            results["metrics"]["base"] = base_metrics

            self.stdout.write(self.style.SUCCESS("Base model results:"))
            self._print_metrics(base_metrics)

        # Comparison
        if "finetuned" in results["metrics"] and "base" in results["metrics"]:
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS("Comparison:"))
            ft = results["metrics"]["finetuned"]
            base = results["metrics"]["base"]

            acc_diff = ft.get("accuracy", 0) - base.get("accuracy", 0)
            self.stdout.write(
                f"  Accuracy: {'+' if acc_diff >= 0 else ''}{acc_diff*100:.1f}%"
            )

        # Save results
        if options["output"]:
            with open(options["output"], "w") as f:
                json.dump(results, f, indent=2)
            self.stdout.write(f"\nResults saved to: {options['output']}")

    def _evaluate_model(self, trainer, test_pairs) -> dict:
        """Evaluate model on test pairs."""
        correct = 0
        total = 0

        for pair in test_pairs:
            # Generate response for prompt
            # This is a simplified evaluation - just checks if chosen is preferred
            prompt = pair.prompt

            try:
                # Get model's preference
                inputs = trainer.tokenizer(
                    prompt + "\n" + pair.chosen,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                chosen_loss = trainer.model(**inputs.to(trainer.model.device), labels=inputs["input_ids"]).loss.item()

                inputs = trainer.tokenizer(
                    prompt + "\n" + pair.rejected,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                rejected_loss = trainer.model(**inputs.to(trainer.model.device), labels=inputs["input_ids"]).loss.item()

                # Lower loss = higher preference
                if chosen_loss < rejected_loss:
                    correct += 1
                total += 1

            except Exception as e:
                self.stdout.write(self.style.WARNING(f"  Error evaluating sample: {e}"))
                continue

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    def _print_metrics(self, metrics: dict):
        """Print metrics in a formatted way."""
        self.stdout.write(f"  Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
        self.stdout.write(f"  Correct: {metrics.get('correct', 0)}/{metrics.get('total', 0)}")
