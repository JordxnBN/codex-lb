from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.core.auth import DEFAULT_PLAN
from app.core.auth.refresh import RefreshError, refresh_access_token, should_refresh
from app.core.balancer import PERMANENT_FAILURE_CODES
from app.core.crypto import TokenEncryptor
from app.core.plan_types import coerce_account_plan_type
from app.core.utils.time import utcnow
from app.db.models import Account, AccountStatus


class AccountsRepositoryPort(Protocol):
    async def update_status(
        self,
        account_id: str,
        status: AccountStatus,
        deactivation_reason: str | None = None,
    ) -> bool: ...

    async def update_tokens(
        self,
        account_id: str,
        access_token_encrypted: bytes,
        refresh_token_encrypted: bytes,
        id_token_encrypted: bytes,
        last_refresh: datetime,
        plan_type: str | None = None,
        email: str | None = None,
    ) -> bool: ...


class AuthManager:
    def __init__(self, repo: AccountsRepositoryPort) -> None:
        self._repo = repo
        self._encryptor = TokenEncryptor()

    async def ensure_fresh(self, account: Account, *, force: bool = False) -> Account:
        if force or should_refresh(account.last_refresh):
            return await self.refresh_account(account)
        return account

    async def refresh_account(self, account: Account) -> Account:
        refresh_token = self._encryptor.decrypt(account.refresh_token_encrypted)
        try:
            result = await refresh_access_token(refresh_token)
        except RefreshError as exc:
            if exc.is_permanent:
                reason = PERMANENT_FAILURE_CODES.get(exc.code, exc.message)
                await self._repo.update_status(account.id, AccountStatus.DEACTIVATED, reason)
                account.status = AccountStatus.DEACTIVATED
                account.deactivation_reason = reason
            raise

        account.access_token_encrypted = self._encryptor.encrypt(result.access_token)
        account.refresh_token_encrypted = self._encryptor.encrypt(result.refresh_token)
        account.id_token_encrypted = self._encryptor.encrypt(result.id_token)
        account.last_refresh = utcnow()
        if result.plan_type is not None:
            account.plan_type = coerce_account_plan_type(
                result.plan_type,
                account.plan_type or DEFAULT_PLAN,
            )
        elif not account.plan_type:
            account.plan_type = DEFAULT_PLAN
        if result.email:
            account.email = result.email

        await self._repo.update_tokens(
            account.id,
            access_token_encrypted=account.access_token_encrypted,
            refresh_token_encrypted=account.refresh_token_encrypted,
            id_token_encrypted=account.id_token_encrypted,
            last_refresh=account.last_refresh,
            plan_type=account.plan_type,
            email=account.email,
        )
        return account
