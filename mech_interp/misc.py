from collections import defaultdict


def construct_pairs(data, concept):
    """Pair up each sample's correct_trigger and clean versions for a concept."""

    lookup = defaultdict(dict)

    for entry in data:
        lookup[entry["id"]][entry["type"]] = entry

    return [
        (group["correct_trigger"], group["clean"])
        for group in lookup.values()
        if "clean" in group
        and "correct_trigger" in group
        and group["clean"]["concept"] == concept
    ]
