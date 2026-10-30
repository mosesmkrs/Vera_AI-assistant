"""Create the first administrative account for local/deployed bootstrap."""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth.security import hash_password
from app.config import settings
from app.db.models import AdminUser


async def create_admin(name: str, email: str, password: str) -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            existing = await session.scalar(select(AdminUser).where(AdminUser.email == email))
            if existing is not None:
                raise ValueError(f"An account with email {email} already exists.")
            session.add(
                AdminUser(
                    name=name,
                    email=email,
                    password_hash=hash_password(password),
                    role="admin",
                )
            )
            await session.commit()
            print(f"Created admin account for {email}.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        parser.error("Passwords do not match.")
    asyncio.run(create_admin(args.name.strip(), args.email.strip().casefold(), password))


if __name__ == "__main__":
    main()
