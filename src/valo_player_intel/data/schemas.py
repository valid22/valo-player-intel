from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


Cohort = Literal["pro", "public"]


class MatchRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    match_id: str
    match_datetime: datetime | None = None
    cohort: Cohort
    source_name: str
    map_name: str | None = None
    team_a_id: str
    team_b_id: str
    winning_team_id: str
    best_of: int | None = None
    event_tier: str | None = None


class PlayerMatchRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    match_id: str
    player_id: str
    player_name: str
    team_id: str
    agent: str | None = None
    agent_role: str | None = None
    kills: float | None = None
    deaths: float | None = None
    assists: float | None = None
    headshot_rate: float | None = None
    damage: float | None = None
    adr: float | None = None
    acs: float | None = None
    kast: float | None = None
    first_kills: float | None = None
    first_deaths: float | None = None
    plants: float | None = None
    defuses: float | None = None
    econ_spend: float | None = None
    econ_value: float | None = None
    utility_used: float | None = None
    utility_damage: float | None = None
    won_match: int


MATCH_COLUMNS = list(MatchRecord.model_fields.keys())
PLAYER_MATCH_COLUMNS = list(PlayerMatchRecord.model_fields.keys())
