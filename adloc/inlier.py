"""
Inlier picks added to located events, as VORA links stations (see utils.invert: picks with a true "candidate" column).

The station graph is VORA's, as its association step writes it: a "neighbors" column in the stations table, the
station_ids each station links to (the links need not be mutual).

select_candidates  the picks each located event may take: at its positive stations and all their neighbours, from
                   its origin time to the latest arrival (P or S) at the furthest of those stations
keep_best_added    a pick added to more than one event stays only with the one it fits best
"""

import numpy as np
import pandas as pd

from .eikonal2d import calc_traveltime

EARTH_RADIUS_KM = 6371.0


def great_circle_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def select_candidates(events, event_picks, pool, stations, config, event_key="idx_eve"):
    """
    The candidate picks of each event, for utils.invert: the pool's picks at the event's positive stations and all
    their neighbours, between its origin time and the latest arrival (P or S, from its location) at the furthest of
    those stations, both widened by max_residual_time. A pool pick can be a candidate of several events.
    events: the events' locations, columns event_key, longitude, latitude, depth_km, time
    event_picks: the events' picks, columns event_key, station_id (and mask: positive stations are those of its
        inliers, mask 1; without mask, of all its picks)
    pool: the picks that may be added (e.g. those of no event), columns station_id, phase_type, phase_time, ...
    stations: columns station_id, longitude, latitude, z_km, neighbors (the station_ids each station links to)
    config: eikonal (travel-time tables, or None for straight rays at config["vel"]), max_residual_time
    Returns the candidate picks: the pool's columns, event_key and candidate (True).
    """
    positive = event_picks[event_picks["mask"] == 1] if "mask" in event_picks.columns else event_picks
    positive = positive[[event_key, "station_id"]].drop_duplicates()
    links = stations[["station_id", "neighbors"]].explode("neighbors").dropna()
    linked = positive.merge(links, on="station_id")[[event_key, "neighbors"]].rename(columns={"neighbors": "station_id"})
    pairs = pd.concat([positive, linked]).drop_duplicates()
    pairs = pairs[pairs["station_id"].isin(pool["station_id"])]
    pairs = pairs.merge(events[[event_key, "longitude", "latitude", "depth_km", "time"]], on=event_key)
    pairs = pairs.merge(stations[["station_id", "longitude", "latitude", "z_km"]], on="station_id", suffixes=("", "_sta"))
    if len(pairs) == 0:
        return pool.iloc[:0].assign(**{event_key: pd.Series(dtype=events[event_key].dtype), "candidate": True})

    # latest arrival at the furthest station of each event: the event at (0, 0, depth), the station at (distance, 0, z)
    distance_km = great_circle_km(pairs["longitude"], pairs["latitude"], pairs["longitude_sta"], pairs["latitude_sta"])
    n = len(pairs)
    event_locs = np.column_stack([np.zeros(n), np.zeros(n), pairs["depth_km"].to_numpy(float)])
    station_locs = np.column_stack([np.asarray(distance_km, float), np.zeros(n), pairs["z_km"].to_numpy(float)])
    vel = config.get("vel", {0: 6.0, 1: 6.0 / 1.73})
    tt = np.maximum(*(calc_traveltime(event_locs, station_locs, np.full(n, phase), config["eikonal"], vel)
                      for phase in (0, 1)))
    margin = pd.Timedelta(seconds=config["max_residual_time"])
    window = pairs.assign(tt=tt).groupby(event_key).agg(time=("time", "first"), tt=("tt", "max"))
    pairs = pairs[[event_key, "station_id"]].merge(window, left_on=event_key, right_index=True)
    pairs["start"] = pairs["time"] - margin
    pairs["end"] = pairs["time"] + pd.to_timedelta(pairs["tt"], unit="s") + margin

    # the pool's picks of each station within each of its events' windows
    pool = pool.reset_index(drop=True)
    rows, keys = [], []
    for station_id, station_pairs in pairs.groupby("station_id"):
        at_station = np.flatnonzero(pool["station_id"].to_numpy() == station_id)
        at_station = at_station[np.argsort(pool["phase_time"].to_numpy()[at_station], kind="stable")]
        times = pool["phase_time"].to_numpy()[at_station]
        lo = np.searchsorted(times, station_pairs["start"].to_numpy(times.dtype), side="left")
        hi = np.searchsorted(times, station_pairs["end"].to_numpy(times.dtype), side="right")
        rows.append(np.concatenate([at_station[a:b] for a, b in zip(lo, hi)] + [np.array([], int)]))
        keys.append(np.repeat(station_pairs[event_key].to_numpy(), hi - lo))
    candidates = pool.iloc[np.concatenate(rows)].copy()
    candidates[event_key] = np.concatenate(keys)
    candidates["candidate"] = True
    return candidates.reset_index(drop=True)


def keep_best_added(picks):
    """
    A pick (station_id, phase_type, phase_time) added to more than one event (candidate and mask 1) stays only with
    the event it fits best, the smallest |residual_time|; its other copies are dropped.
    """
    if "candidate" not in picks.columns:
        return picks
    added = np.flatnonzero(picks["candidate"].fillna(False).to_numpy(bool) & (picks["mask"].to_numpy() == 1))
    order = added[np.argsort(np.abs(picks["residual_time"].to_numpy()[added]), kind="stable")]
    repeated = picks.iloc[order][["station_id", "phase_type", "phase_time"]].duplicated().to_numpy()
    keep = np.ones(len(picks), dtype=bool)
    keep[order[repeated]] = False
    return picks[keep]
