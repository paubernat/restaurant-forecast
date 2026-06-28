"""Demand forecasting for hospitality — hexagonal core.

Layers:
  domain/       pure logic — entities, features, evaluation, validation, plus its own
                ports/ (Protocol interfaces) and services/ (orchestration). No IO.
  adapters/     concrete implementations of the ports (data, models, tracking, storage,
                plotting). The only layer that does IO / imports third-party libs.
  application/  thin use cases: load via an adapter, delegate to a domain service.
  entrypoints/  composition roots (cli) — the only place adapters are wired.
"""

__version__ = "0.1.0"
