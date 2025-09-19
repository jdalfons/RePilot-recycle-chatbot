"""Utility script to seed a local MongoDB instance with the JSON data."""

from __future__ import annotations

import argparse
import logging

from pymongo.errors import ConnectionFailure

from database.db_management import MongoDB


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for MongoDB seeding."""

    parser = argparse.ArgumentParser(
        description="Seed MongoDB with recycling instructions stored as JSON."
    )
    parser.add_argument(
        "--database",
        default="rag",
        help="MongoDB database name (defaults to 'rag').",
    )
    parser.add_argument(
        "--collection",
        default="dechets",
        help="MongoDB collection name to populate (defaults to 'dechets').",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the JSON seed files (defaults to 'data').",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override the MongoDB host (uses MONGO_HOST env var or 'localhost' by default).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the MongoDB port (uses MONGO_PORT env var or 27017 by default).",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection URI. If provided, host/port options are ignored.",
    )
    return parser.parse_args()


def main() -> None:
    """Seed MongoDB using the resolved connection parameters."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    try:
        MongoDB(
            db_name=args.database,
            collection_name=args.collection,
            data_dir=args.data_dir,
            host=args.host,
            port=args.port,
            uri=args.uri,
        )
        logging.info(
            "✅ MongoDB seeding completed for database '%s' and collection '%s'.",
            args.database,
            args.collection,
        )
    except ConnectionFailure as exc:
        logging.error("❌ Unable to connect to MongoDB: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
