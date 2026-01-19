# Changelog

## [0.2.0](https://github.com/Soju06/codex-lb/compare/v0.1.5...v0.2.0) (2026-01-19)


### Features

* add ty type checking and pre-commit hook
* add health response schema and typed context cleanup


### Bug Fixes

* normalize stored plan types (pro/team/business/enterprise/edu) so accounts no longer show as unknown
* prevent rate-limit status when usage is below 100% by using cooldown/backoff and primary-window quota checks
* surface per-account quota reset times by applying primary/secondary reset windows with fallbacks


### Refactor

* move auth/usage helpers into module boundaries and extract proxy helpers
* tighten typing across services and tests

## [0.1.5](https://github.com/Soju06/codex-lb/compare/v0.1.4...v0.1.5) (2026-01-14)


### Bug Fixes

* align rate-limit backoff and reset handling ([4d59650](https://github.com/Soju06/codex-lb/commit/4d596508e5ad13e68aa6e64f9cb32324bd38f07b))

## [0.1.4](https://github.com/Soju06/codex-lb/compare/v0.1.3...v0.1.4) (2026-01-13)


### Bug Fixes

* **db:** harden session cleanup on cancellation ([dee3916](https://github.com/Soju06/codex-lb/commit/dee3916efa83dedec1d5ad43e1e14950b8c6e4a7))

## [0.1.3](https://github.com/Soju06/codex-lb/compare/v0.1.2...v0.1.3) (2026-01-12)


### Documentation

* use absolute image URLs for PyPI ([5fa65a5](https://github.com/Soju06/codex-lb/commit/5fa65a572980f356738f49be3adf2c62fdc38466))

## [0.1.2](https://github.com/Soju06/codex-lb/compare/v0.1.1...v0.1.2) (2026-01-12)


### Bug Fixes

* sync package __version__ ([3dd97e6](https://github.com/Soju06/codex-lb/commit/3dd97e6397a8ea9d3528c166d1e729936f98f737))

## [0.1.1](https://github.com/Soju06/codex-lb/compare/v0.1.0...v0.1.1) (2026-01-12)


### Bug Fixes

* address lint warnings ([7c3cc06](https://github.com/Soju06/codex-lb/commit/7c3cc06c9a6a9a9a8895c1dd5fcc57b3c0eebdb3))
* reactivate accounts when secondary quota clears ([58a4263](https://github.com/Soju06/codex-lb/commit/58a42630d644559f96f045a96c25d0126810542e))
* skip project install in docker build ([64e9156](https://github.com/Soju06/codex-lb/commit/64e9156075c256ef48c0587ea1abb7cc092b97a5))


### Documentation

* add dashboard hero and accounts view ([3522654](https://github.com/Soju06/codex-lb/commit/3522654fe5d09adbe32895d4b24e8b00faac9dfe))

## [0.1.0](https://github.com/Soju06/codex-lb/releases/tag/v0.1.0) (2026-01-07)


### Bug Fixes

* address lint warnings ([7c3cc06](https://github.com/Soju06/codex-lb/commit/7c3cc06c9a6a9a8895c1dd5fcc57b3c0eebdb3))
* skip project install in docker build ([64e9156](https://github.com/Soju06/codex-lb/commit/64e9156075c256ef48c0587ea1abb7cc092b97a5))
