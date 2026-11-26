"""Deterministic query holdouts that keep previously evaluated items in development."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict


def stratified_take(rows, count, *, seed, selected=()):
    """Fill a fixed-size set toward category quotas without moving forced items."""
    by_id = {r["qid"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate query IDs")
    chosen = set(selected)
    if not chosen <= by_id.keys() or not len(chosen) <= count <= len(rows):
        raise ValueError("Invalid holdout size or forced development IDs")
    totals = Counter(str(r.get("category", "")) for r in rows)
    actual = Counter(str(by_id[q].get("category", "")) for q in chosen)
    buckets = defaultdict(list)
    for row in rows:
        if row["qid"] not in chosen:
            buckets[str(row.get("category", ""))].append(row["qid"])
    for values in buckets.values():
        values.sort(key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    while len(chosen) < count:
        category = max((c for c, values in buckets.items() if values),
                       key=lambda c: (count * totals[c] / len(rows) - actual[c], c))
        chosen.add(buckets[category].pop())
        actual[category] += 1
    return chosen


def make_domain_split(rows, exposed, *, dev_size=400, validation_size=200, seed=20261124):
    """Use only IDs/categories/exposure to assign rows, never answers or scores."""
    dev = stratified_take(rows, dev_size, seed=f"{seed}:development", selected=exposed)
    remaining = [r for r in rows if r["qid"] not in dev]
    validation = stratified_take(remaining, validation_size, seed=f"{seed}:validation")
    return {r["qid"]: ("development" if r["qid"] in dev else
                        "validation" if r["qid"] in validation else "test") for r in rows}


def grouped_take(rows, count, *, seed, selected=()):
    """Keep identical-question groups together while filling category deficits."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    by_id = {r["qid"]: r for r in rows}
    if not set(selected) <= by_id.keys() or len(by_id) != len(rows):
        raise ValueError("Invalid forced IDs or duplicate query IDs")
    chosen_groups = {by_id[q]["group"] for q in selected}
    chosen = {r["qid"] for g in chosen_groups for r in groups[g]}
    if not len(chosen) <= count <= len(rows):
        raise ValueError("Forced group closure exceeds the requested split size")
    totals = Counter(str(r.get("category", "")) for r in rows)
    actual = Counter(str(by_id[q].get("category", "")) for q in chosen)
    while len(chosen) < count:
        available = [g for g, members in groups.items() if g not in chosen_groups and
                     len(members) <= count-len(chosen)]
        if not available:
            raise ValueError("Group sizes cannot fill the requested split exactly")
        def priority(group):
            members = groups[group]
            category_counts = Counter(str(r.get("category", "")) for r in members)
            deficit = sum(n * (count*totals[c]/len(rows) - actual[c])
                          for c, n in sorted(category_counts.items())) / len(members)
            return deficit, hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
        group = max(available, key=priority)
        chosen_groups.add(group)
        for row in groups[group]:
            chosen.add(row["qid"])
            actual[str(row.get("category", ""))] += 1
    return chosen


def make_grouped_split(rows, exposed, *, dev_size=600, validation_size=200, seed=20261124):
    dev = grouped_take(rows, dev_size, seed=f"{seed}:development", selected=exposed)
    remaining = [r for r in rows if r["qid"] not in dev]
    validation = grouped_take(remaining, validation_size, seed=f"{seed}:validation")
    return {r["qid"]: ("development" if r["qid"] in dev else
                        "validation" if r["qid"] in validation else "test") for r in rows}
