from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import DEFAULT_PLAN
from app.core.plan_types import coerce_account_plan_type
from app.db.models import Account


async def run(session: AsyncSession) -> None:
    result = await session.execute(select(Account))
    accounts = list(result.scalars().all())
    for account in accounts:
        coerced = coerce_account_plan_type(account.plan_type, DEFAULT_PLAN)
        if account.plan_type != coerced:
            account.plan_type = coerced
