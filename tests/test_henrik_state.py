import pandas as pd

from valo_player_intel.data.sources.henrik import _merge_seed_queue, _player_key


def test_merge_seed_queue_skips_completed_pending_and_failed():
    seeds = pd.DataFrame(
        [
            {"region": "na", "name": "Alpha", "tag": "ONE"},
            {"region": "na", "name": "Bravo", "tag": "TWO"},
            {"region": "eu", "name": "Charlie", "tag": "THREE"},
        ]
    )
    state = {
        "pending_players": [{"region": "na", "name": "Alpha", "tag": "ONE"}],
        "completed_players": [_player_key("na", "Bravo", "TWO")],
        "failed_players": {_player_key("eu", "Charlie", "THREE"): 3},
        "last_seed_count": 0,
    }

    queue = _merge_seed_queue(state, seeds)

    assert queue == [{"region": "na", "name": "Alpha", "tag": "ONE"}]
