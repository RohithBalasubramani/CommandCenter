"""
Management command to index industrial equipment data for RAG.

Usage:
    python manage.py index_rag           # Index all data
    python manage.py index_rag --clear   # Clear and re-index
    python manage.py index_rag --stats   # Show index stats
"""

from django.core.management.base import BaseCommand
from layer2.rag_pipeline import get_rag_pipeline


class Command(BaseCommand):
    help = "Index industrial equipment data for RAG retrieval"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear existing index before re-indexing",
        )
        parser.add_argument(
            "--stats",
            action="store_true",
            help="Show index statistics only",
        )

    def handle(self, *args, **options):
        pipeline = get_rag_pipeline()

        if options["stats"]:
            self.show_stats(pipeline)
            return

        if options["clear"]:
            self.stdout.write("Clearing existing index...")
            pipeline.clear_index()

        self.stdout.write("Indexing equipment data...")
        pipeline.index_equipment_from_db()

        self.stdout.write(self.style.SUCCESS("Indexing complete!"))
        self.show_stats(pipeline)

    def show_stats(self, pipeline):
        """Show index statistics."""
        stats = pipeline.get_stats()

        total = sum(v for k, v in stats.items() if k.endswith("_count"))

        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("RAG INDEX STATISTICS")
        self.stdout.write("=" * 50)
        self.stdout.write(f"  Equipment documents:    {stats['equipment_count']}")
        self.stdout.write(f"  Alert documents:        {stats['alerts_count']}")
        self.stdout.write(f"  Maintenance documents:  {stats['maintenance_count']}")
        self.stdout.write(f"  Operational documents:  {stats.get('operational_docs_count', 0)}")
        self.stdout.write(f"  Shift logs:             {stats.get('shift_logs_count', 0)}")
        self.stdout.write(f"  Work orders:            {stats.get('work_orders_count', 0)}")
        self.stdout.write(f"  ---")
        self.stdout.write(f"  TOTAL indexed docs:     {total}")
        self.stdout.write(f"  LLM available:          {stats['llm_available']}")
        self.stdout.write(f"  LLM model:              {stats['llm_model']}")
        self.stdout.write("=" * 50)
