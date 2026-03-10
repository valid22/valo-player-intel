from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from valo_player_intel.data.io import read_json, write_json


BASE_URL = "https://vlrggapi.vercel.app/v2"
OFFICIAL_EVENT_KEYWORDS = (
    "VCT",
    "MASTERS",
    "CHAMPIONS",
    "CHALLENGERS",
    "ASCENSION",
    "GAME CHANGERS",
    "PACIFIC",
    "AMERICAS",
    "EMEA",
    "CN",
)


@dataclass(slots=True)
class VLRConfig:
    pages: int = 3
    completed_limit_per_event: int = 20
    timeout_seconds: float = 20.0
    append: bool = True


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise requests.HTTPError(str(payload))
    return payload


def _extract_id_from_url(url: str | None) -> int | None:
    if not url:
        return None
    try:
        parts = [part for part in url.split("/") if part]
        event_index = parts.index("event")
        return int(parts[event_index + 1])
    except Exception:
        return None


def fetch_vct_matches(config: VLRConfig, raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    events_path = raw_dir / "vlr_events.json"
    matches_path = raw_dir / "vlr_matches.json"
    players_path = raw_dir / "vlr_player_matches.json"
    details_path = raw_dir / "vlr_match_details.json"

    existing_events = read_json(events_path, default={"events": []}).get("events", []) if config.append else []
    existing_matches = read_json(matches_path, default={"matches": []}).get("matches", []) if config.append else []
    existing_players = read_json(players_path, default={"player_matches": []}).get("player_matches", []) if config.append else []
    existing_details = read_json(details_path, default={"match_details": []}).get("match_details", []) if config.append else []

    raw_events: list[dict[str, Any]] = list(existing_events)
    raw_match_details: list[dict[str, Any]] = list(existing_details)
    match_rows: list[dict[str, Any]] = list(existing_matches)
    player_rows: list[dict[str, Any]] = list(existing_players)
    seen_match_ids: set[int] = {int(row["match_id"]) for row in existing_matches if row.get("match_id")}
    seen_event_ids: set[int] = {_extract_id_from_url(row.get("url_path")) for row in existing_events if row.get("url_path")}
    seen_event_ids.discard(None)
    seen_player_keys: set[tuple[str, str]] = {
        (row.get("match_id"), row.get("player_id")) for row in existing_players if row.get("match_id") and row.get("player_id")
    }
    seen_detail_ids: set[str] = set()
    for row in existing_details:
        try:
            seen_detail_ids.add(str(row["segments"][0]["match_id"]))
        except Exception:
            continue

    for page in range(1, config.pages + 1):
        payload = _get_json("/events", {"q": "completed", "page": page})
        events = payload.get("data", {}).get("segments", [])
        for event in events:
            event_id = _extract_id_from_url(event.get("url_path"))
            title = str(event.get("title", ""))
            if event_id is None or event_id in seen_event_ids or not _is_official_vct_event(title):
                continue
            seen_event_ids.add(event_id)
            raw_events.append(event)

            matches_payload = _get_json("/events/matches", {"event_id": event_id})
            matches = matches_payload.get("data", {}).get("segments", [])
            fetched_for_event = 0
            for match_stub in matches:
                match_id = int(match_stub.get("match_id") or 0)
                if not match_id or match_id in seen_match_ids:
                    continue
                seen_match_ids.add(match_id)
                details_payload = _get_json("/match/details", {"match_id": match_id})
                details = details_payload.get("data", {})
                parsed = _parse_match_details(match_id, details, event)
                if parsed is None:
                    continue
                if str(match_id) not in seen_detail_ids:
                    raw_match_details.append(details)
                    seen_detail_ids.add(str(match_id))
                match_rows.append(parsed["match"])
                for row in parsed["player_matches"]:
                    key = (row["match_id"], row["player_id"])
                    if key in seen_player_keys:
                        continue
                    seen_player_keys.add(key)
                    player_rows.append(row)
                fetched_for_event += 1
                if fetched_for_event >= config.completed_limit_per_event:
                    break

    write_json({"events": raw_events}, events_path)
    write_json({"matches": match_rows}, matches_path)
    write_json({"player_matches": player_rows}, players_path)
    write_json({"match_details": raw_match_details}, details_path)
    return {
        "events": events_path,
        "matches": matches_path,
        "player_matches": players_path,
        "match_details": details_path,
    }


def _parse_match_details(match_id: int, details: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    segments = details.get("segments", [])
    if not segments:
        return None
    segment = segments[0]
    teams = segment.get("teams", [])
    maps = segment.get("maps", [])
    if len(teams) < 2 or not maps:
        return None

    team_a = teams[0]
    team_b = teams[1]
    team_a_id = team_a.get("name")
    team_b_id = team_b.get("name")
    winning_team_id = team_a_id if int(team_a.get("score") or 0) >= int(team_b.get("score") or 0) else team_b_id

    first_map = maps[0]
    match_row = {
        "match_id": str(match_id),
        "match_datetime": _parse_match_datetime(segment.get("date"), event),
        "cohort": "pro",
        "source_name": "vlr_vct_public",
        "map_name": first_map.get("map_name"),
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "winning_team_id": winning_team_id,
        "best_of": _best_of_from_scores(team_a.get("score"), team_b.get("score"), len(maps)),
        "event_tier": event.get("title"),
    }

    player_rows = _aggregate_players(str(match_id), maps, winning_team_id, team_a_id, team_b_id)
    if not player_rows:
        return None
    return {"match": match_row, "player_matches": player_rows}


def _aggregate_players(
    match_id: str,
    maps: list[dict[str, Any]],
    winning_team_id: str,
    team_a_id: str,
    team_b_id: str,
) -> list[dict[str, Any]]:
    players: dict[tuple[str, str], dict[str, Any]] = {}
    map_counts: dict[tuple[str, str], int] = {}
    for map_payload in maps:
        for team_slot in ("team1", "team2"):
            for player in map_payload.get("players", {}).get(team_slot, []):
                team_id = team_a_id if team_slot == "team1" else team_b_id
                player_name = player.get("name")
                if not team_id or not player_name:
                    continue
                key = (team_id, player_name)
                current = players.setdefault(
                    key,
                    {
                        "match_id": match_id,
                        "player_id": f"{team_id}:{player_name}",
                        "player_name": player_name,
                        "team_id": team_id,
                        "agent": player.get("agent"),
                        "agent_role": pd.NA,
                        "kills": 0.0,
                        "deaths": 0.0,
                        "assists": 0.0,
                        "headshot_rate": None,
                        "damage": None,
                        "adr": 0.0,
                        "acs": 0.0,
                        "kast": 0.0,
                        "first_kills": 0.0,
                        "first_deaths": 0.0,
                        "plants": None,
                        "defuses": None,
                        "econ_spend": None,
                        "econ_value": _to_float(player.get("rating")),
                        "utility_used": None,
                        "utility_damage": None,
                        "won_match": int(team_id == winning_team_id),
                        "cohort": "pro",
                    },
                )
                map_counts[key] = map_counts.get(key, 0) + 1
                current["kills"] += _to_float(player.get("kills"))
                current["deaths"] += _to_float(player.get("deaths"))
                current["assists"] += _to_float(player.get("assists"))
                current["adr"] += _to_float(player.get("adr"))
                current["acs"] += _to_float(player.get("acs"))
                current["kast"] += _percent_to_float(player.get("kast")) or 0.0
                current["first_kills"] += _to_float(player.get("fk"))
                current["first_deaths"] += _to_float(player.get("fd"))
                hs = _percent_to_float(player.get("hs_pct"))
                current["headshot_rate"] = hs if current["headshot_rate"] is None else (current["headshot_rate"] + hs) / 2 if hs is not None else current["headshot_rate"]
                if not current["agent"] and player.get("agent"):
                    current["agent"] = player.get("agent")

    output = []
    for key, row in players.items():
        count = max(1, map_counts[key])
        row["adr"] = row["adr"] / count if row["adr"] is not None else None
        row["acs"] = row["acs"] / count if row["acs"] is not None else None
        row["kast"] = row["kast"] / count if row["kast"] is not None else None
        output.append(row)
    return output


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _percent_to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    number = float(text)
    return number / 100 if number > 1 else number


def _best_of_from_scores(score_a: Any, score_b: Any, map_count: int) -> int:
    try:
        return max(1, int(score_a or 0) + int(score_b or 0))
    except Exception:
        return map_count or 1


def _is_official_vct_event(title: str) -> bool:
    upper = title.upper()
    return any(keyword in upper for keyword in OFFICIAL_EVENT_KEYWORDS)


def _parse_match_datetime(raw_date: Any, event: dict[str, Any]) -> str | None:
    if raw_date is None:
        return None
    text = str(raw_date).strip()
    if not text:
        return None
    text = re.sub(r"Patch\s+\d+(\.\d+)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"([A-Za-z]+)\s+(\d{1,2})(\d{1,2}:\d{2}\s*[AP]M)", r"\1 \2 \3", text)
    text = re.sub(r"\b(?:EST|EDT|CST|CDT|PST|PDT|GMT|UTC)\b", "", text).strip()
    year_match = re.search(r"(20\d{2})", str(event.get("title", "")))
    year = year_match.group(1) if year_match else None
    if year and year not in text:
        text = f"{text} {year}"
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.isoformat()
