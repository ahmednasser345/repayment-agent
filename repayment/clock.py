"""Simulated time: the demo moves it forward by hand ("Let a day pass"); nothing else does."""

from datetime import datetime, timedelta, timezone


class SimClock:
    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("SimClock needs a timezone-aware start time")
        self._now = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, **delta: float) -> None:
        """advance(hours=24), advance(minutes=15), ..."""
        self._now += timedelta(**delta)


def current_minute() -> datetime:
    """The real time in UTC, rounded down to the minute: where the app's clock starts."""
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)
