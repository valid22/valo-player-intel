from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from valo_player_intel.data.io import read_json, write_json


BASE_URL = "https://api.henrikdev.xyz/valorant"


@dataclass(slots=True)
class HenrikConfig:
    regions: tuple[str, ...] = ("na", "eu", "ap", "kr", "latam", "br")
    leaderboard_size: int = 50
    leaderboard_start_indices: tuple[int, ...] = (0, 50, 100, 150, 200)
    matches_per_player: int = 10
    seed_players_path: Path | None = None
    request_pause_seconds: float = 2.1
    append: bool = True
    max_players_per_run: int = 25
    discover_from_leaderboard: bool = True
    recycle_completed_when_empty: int = 10


class HenrikRateLimitError(requests.HTTPError):
    """Raised when Henrik responds with a rate-limit error."""


def _headers() -> dict[str, str]:
    token = os.getenv("HENRIK_API_KEY", "").strip()
    return {"Authorization": token} if token else {}


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((requests.HTTPError, HenrikRateLimitError)),
)
def _get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(f"{BASE_URL}{path}", params=params or {}, headers=_headers(), timeout=30)
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                time.sleep(max(float(retry_after), 1.0))
            except ValueError:
                time.sleep(5.0)
        raise HenrikRateLimitError("Henrik API rate limit hit (429).")
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") not in (200, None):
        raise requests.HTTPError(f"Henrik API returned status {payload.get('status')}: {payload}")
    return payload


def _load_seed_players(config: HenrikConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if config.seed_players_path and config.seed_players_path.exists():
        frames.append(pd.read_csv(config.seed_players_path))

    if not config.discover_from_leaderboard:
        if not frames:
            return pd.DataFrame(columns=["name", "tag", "region"])
        seeds = pd.concat(frames, ignore_index=True)
        if "gameName" in seeds.columns and "name" not in seeds.columns:
            seeds["name"] = seeds["gameName"]
        if "tagLine" in seeds.columns and "tag" not in seeds.columns:
            seeds["tag"] = seeds["tagLine"]
        required = ["name", "tag", "region"]
        for column in required:
            if column not in seeds.columns:
                seeds[column] = pd.NA
        seeds = seeds[required].dropna(subset=["name", "tag"]).copy()
        seeds["name"] = seeds["name"].astype(str).str.strip()
        seeds["tag"] = seeds["tag"].astype(str).str.strip()
        seeds = seeds[(seeds["name"] != "") & (seeds["tag"] != "")]
        return seeds.drop_duplicates()

    for region in config.regions:
        for start_index in config.leaderboard_start_indices:
            payload = _get_json(
                f"/v3/leaderboard/{region}/pc",
                params={"size": config.leaderboard_size, "startIndex": start_index},
            )
            data = payload.get("data", {})
            players = data.get("players", []) if isinstance(data, dict) else payload.get("data", [])
            if not players:
                continue
            frame = pd.DataFrame(players)
            frame["region"] = region
            frame["start_index"] = start_index
            frames.append(frame)
            time.sleep(config.request_pause_seconds)

    if not frames:
        return pd.DataFrame(columns=["name", "tag", "region"])

    seeds = pd.concat(frames, ignore_index=True)
    if "gameName" in seeds.columns and "name" not in seeds.columns:
        seeds["name"] = seeds["gameName"]
    if "tagLine" in seeds.columns and "tag" not in seeds.columns:
        seeds["tag"] = seeds["tagLine"]
    required = ["name", "tag", "region"]
    for column in required:
        if column not in seeds.columns:
            seeds[column] = pd.NA
    seeds = seeds[required].dropna(subset=["name", "tag"]).copy()
    seeds["name"] = seeds["name"].astype(str).str.strip()
    seeds["tag"] = seeds["tag"].astype(str).str.strip()
    seeds = seeds[(seeds["name"] != "") & (seeds["tag"] != "")]
    return seeds.drop_duplicates()


def _extract_team(team_payload: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    return {
        "team_id": team_payload.get("team_id") or team_payload.get("name") or fallback_name,
        "team_name": team_payload.get("name") or fallback_name,
        "rounds_won": team_payload.get("rounds_won", team_payload.get("rounds", {}).get("won")),
        "rounds_lost": team_payload.get("rounds_lost", team_payload.get("rounds", {}).get("lost")),
    }


def _player_key(region: str, name: str, tag: str) -> str:
    return f"{region}|{name}|{tag}"


def _load_crawl_state(state_path: Path) -> dict[str, Any]:
    state = read_json(
        state_path,
        default={"pending_players": [], "completed_players": [], "failed_players": {}, "last_seed_count": 0},
    )
    state.setdefault("pending_players", [])
    state.setdefault("completed_players", [])
    state.setdefault("failed_players", {})
    state.setdefault("last_seed_count", 0)
    return state


def _merge_seed_queue(state: dict[str, Any], seeds: pd.DataFrame) -> list[dict[str, str]]:
    pending = list(state["pending_players"])
    completed = set(state["completed_players"])
    failed = state["failed_players"]
    pending_keys = {_player_key(item["region"], item["name"], item["tag"]) for item in pending}

    for row in seeds.itertuples(index=False):
        item = {"region": str(row.region), "name": str(row.name), "tag": str(row.tag)}
        key = _player_key(item["region"], item["name"], item["tag"])
        if key in completed or key in pending_keys:
            continue
        failure_count = int(failed.get(key, 0))
        if failure_count >= 3:
            continue
        pending.append(item)
        pending_keys.add(key)
    return pending


def _completed_key_to_item(key: str) -> dict[str, str] | None:
    parts = key.split("|", 2)
    if len(parts) != 3:
        return None
    region, name, tag = parts
    if not region or not name or not tag:
        return None
    return {"region": region, "name": name, "tag": tag}


def fetch_public_matches(config: HenrikConfig, raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    matches_path = raw_dir / "henrik_matches.json"
    players_path = raw_dir / "henrik_player_matches.json"
    seed_path = raw_dir / "public_seed_players.csv"
    state_path = raw_dir / "henrik_crawl_state.json"

    existing_matches_payload = read_json(matches_path, default={"matches": []}) if config.append else {"matches": []}
    existing_players_payload = read_json(players_path, default={"player_matches": []}) if config.append else {"player_matches": []}
    existing_matches = existing_matches_payload.get("matches", [])
    existing_player_rows = existing_players_payload.get("player_matches", [])
    seen_match_ids: set[str] = {row.get("match_id") for row in existing_matches if row.get("match_id")}
    seen_player_keys: set[tuple[str, str]] = {
        (row.get("match_id"), row.get("player_id")) for row in existing_player_rows if row.get("match_id") and row.get("player_id")
    }

    seeds = _load_seed_players(config)
    seeds.to_csv(seed_path, index=False)
    state = (
        _load_crawl_state(state_path)
        if config.append
        else {"pending_players": [], "completed_players": [], "failed_players": {}, "last_seed_count": 0}
    )
    pending_players = _merge_seed_queue(state, seeds)
    if not pending_players and config.recycle_completed_when_empty > 0:
        recycled = 0
        for key in reversed(state["completed_players"]):
            item = _completed_key_to_item(key)
            if item is None:
                continue
            pending_players.append(item)
            recycled += 1
            if recycled >= config.recycle_completed_when_empty:
                break

    matches: list[dict[str, Any]] = list(existing_matches)
    player_rows: list[dict[str, Any]] = list(existing_player_rows)
    processed_players = 0
    completed_players = set(state["completed_players"]) if config.append else set()
    failed_players: dict[str, int] = dict(state["failed_players"]) if config.append else {}
    remaining_queue: list[dict[str, str]] = []
    remaining_keys = {_player_key(item["region"], item["name"], item["tag"]) for item in remaining_queue}

    for item in pending_players:
        if processed_players >= config.max_players_per_run:
            remaining_queue.append(item)
            continue
        player_id_key = _player_key(item["region"], item["name"], item["tag"])
        try:
            payload = _get_json(
                f"/v3/matches/{item['region']}/{quote(item['name'], safe='')}/{quote(item['tag'], safe='')}",
                params={"size": config.matches_per_player, "filter": "competitive"},
            )
        except Exception:
            failed_players[player_id_key] = failed_players.get(player_id_key, 0) + 1
            if failed_players[player_id_key] < 3:
                remaining_queue.append(item)
            continue
        processed_players += 1
        completed_players.add(player_id_key)
        failed_players.pop(player_id_key, None)
        time.sleep(config.request_pause_seconds)
        for match in payload.get("data", []) or []:
            if not isinstance(match, dict):
                continue
            metadata = match.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            teams = match.get("teams", {}) or {}
            if not isinstance(teams, dict):
                teams = {}
            blue = _extract_team(teams.get("blue", {}), "Blue")
            red = _extract_team(teams.get("red", {}), "Red")
            match_id = metadata.get("matchid")
            if not match_id:
                continue
            winning_team = _winning_team_label(teams)
            if match_id not in seen_match_ids:
                seen_match_ids.add(match_id)
                matches.append(
                    {
                        "match_id": match_id,
                        "match_datetime": metadata.get("game_start"),
                        "cohort": "public",
                        "source_name": "henrik_public_competitive",
                        "map_name": metadata.get("map"),
                        "team_a_id": blue["team_id"] or "Blue",
                        "team_b_id": red["team_id"] or "Red",
                        "winning_team_id": blue["team_id"] if winning_team == "Blue" else red["team_id"],
                        "best_of": 1,
                        "event_tier": "public_competitive",
                    }
                )

            first_kills, first_deaths = _first_blood_counts(match.get("kills", []) or [])
            plants, defuses = _objective_counts(match.get("rounds", []) or [])
            for player in (match.get("players", {}) or {}).get("all_players", []):
                player_name = str(player.get("name") or "").strip()
                player_tag = str(player.get("tag") or "").strip()
                discovered_key = _player_key(item["region"], player_name, player_tag) if player_name and player_tag else None
                if (
                    discovered_key
                    and discovered_key not in completed_players
                    and failed_players.get(discovered_key, 0) < 3
                    and discovered_key not in remaining_keys
                ):
                    remaining_queue.append({"region": item["region"], "name": player_name, "tag": player_tag})
                    remaining_keys.add(discovered_key)

                stats = player.get("stats", {}) or {}
                damage = player.get("damage_made")
                econ_spend = player.get("economy", {}).get("spent", {}).get("overall")
                econ_value = player.get("economy", {}).get("loadout_value", {}).get("overall")
                player_id = player.get("puuid") or f"{player.get('name')}#{player.get('tag')}"
                player_key = (match_id, player_id)
                if player_key in seen_player_keys:
                    continue
                seen_player_keys.add(player_key)
                team_name = player.get("team")
                player_rows.append(
                    {
                        "match_id": match_id,
                        "player_id": player_id,
                        "player_name": f"{player.get('name', '')}#{player.get('tag', '')}".strip("#"),
                        "team_id": blue["team_id"] if team_name == "Blue" else red["team_id"],
                        "agent": player.get("character"),
                        "agent_role": pd.NA,
                        "kills": stats.get("kills"),
                        "deaths": stats.get("deaths"),
                        "assists": stats.get("assists"),
                        "headshot_rate": _headshot_rate(stats),
                        "damage": damage,
                        "adr": _adr_from_damage(damage, metadata.get("rounds_played")),
                        "acs": pd.NA,
                        "kast": pd.NA,
                        "first_kills": first_kills.get(player_id, 0),
                        "first_deaths": first_deaths.get(player_id, 0),
                        "plants": plants.get(player_id, 0),
                        "defuses": defuses.get(player_id, 0),
                        "econ_spend": econ_spend,
                        "econ_value": econ_value,
                        "utility_used": _ability_total(player.get("ability_casts", {})),
                        "utility_damage": pd.NA,
                        "won_match": int(team_name == winning_team),
                        "cohort": "public",
                    }
                )

    write_json({"matches": matches}, matches_path)
    write_json({"player_matches": player_rows}, players_path)
    write_json(
        {
            "pending_players": remaining_queue,
            "completed_players": sorted(completed_players),
            "failed_players": failed_players,
            "last_seed_count": int(len(seeds)),
            "processed_this_run": processed_players,
        },
        state_path,
    )
    return {"matches": matches_path, "player_matches": players_path, "seed_players": seed_path}


def _headshot_rate(stats: dict[str, Any]) -> float | None:
    headshots = stats.get("headshots")
    bodyshots = stats.get("bodyshots")
    legshots = stats.get("legshots")
    total = sum(x or 0 for x in [headshots, bodyshots, legshots])
    if not total:
        return None
    return (headshots or 0) / total


def _adr_from_damage(damage: float | None, rounds_played: int | None) -> float | None:
    if not damage or not rounds_played:
        return None
    return damage / rounds_played


def _first_blood_counts(kills: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    first_kills: dict[str, int] = {}
    first_deaths: dict[str, int] = {}
    by_round: dict[int, list[dict[str, Any]]] = {}
    for kill in kills:
        by_round.setdefault(int(kill.get("round", -1)), []).append(kill)

    for _, round_kills in by_round.items():
        ordered = sorted(round_kills, key=lambda item: item.get("kill_time_in_round") or 0)
        if not ordered:
            continue
        first = ordered[0]
        killer = first.get("killer_puuid")
        victim = first.get("victim_puuid")
        if killer:
            first_kills[killer] = first_kills.get(killer, 0) + 1
        if victim:
            first_deaths[victim] = first_deaths.get(victim, 0) + 1
    return first_kills, first_deaths


def _objective_counts(rounds: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    plants: dict[str, int] = {}
    defuses: dict[str, int] = {}
    for item in rounds:
        plant = item.get("plant_events") or {}
        planted_by = (plant.get("planted_by") or {}).get("puuid")
        if planted_by:
            plants[planted_by] = plants.get(planted_by, 0) + 1

        defuse = item.get("defuse_events") or {}
        defused_by = (defuse.get("defused_by") or {}).get("puuid")
        if defused_by:
            defuses[defused_by] = defuses.get(defused_by, 0) + 1
    return plants, defuses


def _ability_total(ability_casts: dict[str, Any]) -> int | None:
    if not ability_casts:
        return None
    return int(sum((value or 0) for value in ability_casts.values()))


def _winning_team_label(teams: dict[str, Any]) -> str:
    blue = teams.get("blue", {}) if isinstance(teams, dict) else {}
    red = teams.get("red", {}) if isinstance(teams, dict) else {}
    if blue.get("has_won") is True:
        return "Blue"
    if red.get("has_won") is True:
        return "Red"

    blue_rounds = blue.get("rounds_won")
    red_rounds = red.get("rounds_won")
    if blue_rounds is not None and red_rounds is not None:
        return "Blue" if blue_rounds > red_rounds else "Red"
    return "Blue"
