#!/usr/bin/env python3
import os
import asyncio
from pathlib import Path


async def main():
    try:
        from surrealdb import Surreal  # type: ignore
    except Exception as e:
        raise SystemExit("surrealdb client not installed. `pip install surrealdb`.")

    url = os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    user = os.getenv("SURREALDB_USER", "root")
    pw = os.getenv("SURREALDB_PASS", "root")
    ns = os.getenv("SURREALDB_NAMESPACE", "localcat")
    db = os.getenv("SURREALDB_DATABASE", "memory")

    schema_path = Path(__file__).resolve().parents[1] / "schema" / "surreal" / "unified_knowledge_schema.surql"
    schema = schema_path.read_text(encoding="utf-8")

    client = Surreal(url)
    await client.connect()
    await client.signin({"user": user, "pass": pw})
    await client.use(ns, db)

    for stmt in [s.strip() + ";" for s in schema.split(";") if s.strip()]:
        await client.query(stmt)

    print(f"✅ Schema applied to {ns}/{db}")


if __name__ == "__main__":
    asyncio.run(main())

