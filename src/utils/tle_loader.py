# -*- coding: utf-8 -*-
"""
Reusable TLE loading utilities.

Provides:
- get_tle_data: ID-based fetch with local caching
- get_tle_by_groups_and_name: name-based search from CelesTrak groups (starlink/galileo/active)
- preprocess_satellite_configs: injects TLE tuples (l1, l2) into configs for env init

This module centralizes the TLE logic so both test and training scripts can reuse it.
"""
from __future__ import annotations

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Soft dependency on requests
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    REQUESTS_AVAILABLE = False

# -----------------------------
# ID-based fetch with caching
# -----------------------------

def get_tle_data(norad_id: int, cache_dir: str, cache_duration_hours: int = 24) -> Optional[List[str]]:
    """Fetch TLE by NORAD ID, with local caching.
    Returns a list of non-empty lines from cache or remote, or None on failure.
    
    Fallback behavior: if remote fetch is unavailable or fails, return cached
    file contents even if stale, to keep simulations running offline.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{norad_id}.tle")

    # Helper to read cache (fresh or stale)
    def _read_cache() -> Optional[List[str]]:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return [ln.strip() for ln in f.readlines() if ln.strip()]
            except Exception:
                return None
        return None

    # Use fresh cache if available
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=cache_duration_hours):
            cached = _read_cache()
            if cached:
                return cached

    # Try remote fetch if requests available
    if REQUESTS_AVAILABLE:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
        try:
            resp = requests.get(url, timeout=30, verify=False)
            resp.raise_for_status()
            text = resp.text.strip()
            if text and "No TLE found" not in text:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))
                except Exception:
                    pass
                return lines
        except Exception:
            pass
    else:
        print(f"\n[WARNING] 'requests' not installed. Using cached TLE for {norad_id} if available.")

    # Fallback: return cached file even if stale
    return _read_cache()

# -----------------------------
# Group-based name matching
# -----------------------------

_GROUP_URLS = {
    'starlink': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle',
    'galileo': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle',
    'active':  'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
}


def _normalize_name(s: str) -> str:
    s = (s or '').upper().strip()
    return re.sub(r'[^A-Z0-9]', '', s)


def _fetch_group_text(group: str, groups_cache_dir: str, cache_hours: int = 24) -> Optional[str]:
    if not REQUESTS_AVAILABLE:
        return None
    os.makedirs(groups_cache_dir, exist_ok=True)
    cache_file = os.path.join(groups_cache_dir, f"{group}.tle")

    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(hours=cache_hours):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                pass

    url = _GROUP_URLS.get(group)
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=60, verify=False)
        resp.raise_for_status()
        text = resp.text
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text
    except Exception:
        return None


def _parse_group_tle(text: str) -> List[Tuple[str, str, str]]:
    lines = [ln.strip() for ln in (text or '').splitlines() if ln.strip()]
    triples: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(lines) - 2:
        name_line, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith('1 ') and l2.startswith('2 '):
            triples.append((name_line, l1, l2))
            i += 3
        else:
            i += 1
    return triples


def _match_tle_by_name(target_name: str, groups_texts: Dict[str, str]) -> Optional[Tuple[str, str]]:
    norm_target = _normalize_name(target_name)
    gsat_match = re.search(r'GSAT\d{4}', norm_target)
    gal_num_match = re.search(r'GALILEO(\d+)', norm_target)
    star_match = re.search(r'STARLINK-?(\d+)', (target_name or '').upper())

    for group, text in groups_texts.items():
        if not text:
            continue
        for name_line, l1, l2 in _parse_group_tle(text):
            norm_name = _normalize_name(name_line)
            ok = False
            # Starlink exact number
            if star_match and 'STARLINK' in norm_name:
                code = re.sub(r'\D', '', star_match.group(1))
                ok = _normalize_name(f'STARLINK{code}') in norm_name
            if not ok and gsat_match and 'GSAT' in norm_name:
                ok = gsat_match.group(0) in norm_name
            if not ok and gal_num_match and 'GALILEO' in norm_name:
                ok = _normalize_name(f'GALILEO {gal_num_match.group(1)}') in norm_name
            if not ok and 'GAOFEN' in norm_target and 'GAOFEN' in norm_name:
                ok = norm_target in norm_name or norm_name in norm_target
            if not ok:
                ok = norm_target in norm_name
            if ok:
                return (l1, l2)
    return None


def get_tle_by_groups_and_name(name: str, cache_dir: str) -> Optional[Tuple[str, str]]:
    uname = (name or '').upper()
    groups = []
    if 'STARLINK' in uname:
        groups.append('starlink')
    if 'GALILEO' in uname or 'GSAT' in uname:
        groups.append('galileo')
    if 'GAOFEN' in uname:
        groups.append('active')
    if 'active' not in groups:
        groups.append('active')

    groups_dir = os.path.join(cache_dir, 'groups')
    texts: Dict[str, str] = {}
    for g in groups:
        texts[g] = _fetch_group_text(g, groups_dir) or ''
    return _match_tle_by_name(name, texts)

# -----------------------------
# Public API: preprocess configs
# -----------------------------

def preprocess_satellite_configs(project_root: str, sat_configs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Inject TLE data into the satellite configs.
    Order per satellite:
      1) Use cached file if present and fresh
      2) Try group-based lookup (by name) from starlink/galileo/active and cache
      3) Fallback to ID-based fetch by CATNR
    """
    tle_cache_dir = os.path.join(project_root, 'data', 'tle')
    os.makedirs(tle_cache_dir, exist_ok=True)

    processed = {'source_satellites': [], 'compute_satellites': [], 'leader_satellites': []}
    for group_name, sat_list in sat_configs.items():
        if group_name not in processed:
            continue
        for cfg in sat_list:
            sat_id = int(cfg.get('sat_id'))
            sat_name = cfg.get('name', f'SAT-{sat_id}')

            cached = get_tle_data(sat_id, tle_cache_dir)
            if cached and len(cached) >= 2:
                cfg['tle'] = (cached[-2], cached[-1])
                processed[group_name].append(cfg)
                continue

            tle_pair = get_tle_by_groups_and_name(sat_name, tle_cache_dir)
            if tle_pair:
                l1, l2 = tle_pair
                try:
                    with open(os.path.join(tle_cache_dir, f"{sat_id}.tle"), 'w', encoding='utf-8') as f:
                        f.write(f"{l1}\n{l2}\n")
                except Exception:
                    pass
                cfg['tle'] = (l1, l2)
                processed[group_name].append(cfg)
                continue

            # Fallback again to ID
            cached2 = get_tle_data(sat_id, tle_cache_dir)
            if cached2 and len(cached2) >= 2:
                cfg['tle'] = (cached2[-2], cached2[-1])
                processed[group_name].append(cfg)
            else:
                print(f"\n[WARNING] Skipping satellite {sat_id} ({sat_name}) as its TLE data could not be obtained.")

    return processed

