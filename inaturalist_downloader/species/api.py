"""iNaturalist API helpers for species extraction."""

import json
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import API_BASE, DEFAULT_TIMEOUT, USER_AGENT


def http_get_json(path: str, params: Optional[dict] = None, retries: int = 5) -> dict:
    """Fetch JSON from an iNaturalist API endpoint with retry/backoff."""
    final_url = f"{API_BASE}{path}"
    if params:
        final_url = f"{final_url}?{urlencode(params, doseq=True)}"

    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(final_url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Request failed for {final_url}: {last_error}")


def choose_best_result(results: list[dict], query: str, name_keys: list[str]) -> dict:
    """Choose the best autocomplete result for a user query."""
    if not results:
        raise ValueError(f"No result found for '{query}'")

    normalized_query = query.casefold()
    exact_matches = []
    partial_matches = []

    for item in results:
        values = {str(item.get(key, "")).casefold() for key in name_keys}
        if normalized_query in values:
            exact_matches.append(item)
        elif any(normalized_query in value for value in values if value):
            partial_matches.append(item)

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]
    return results[0]


def resolve_place(place_query: str) -> tuple[int, str]:
    """Resolve a human-readable place name to an iNaturalist place ID."""
    payload = http_get_json("/places/autocomplete", {"q": place_query, "per_page": 10})
    result = choose_best_result(
        payload.get("results", []),
        place_query,
        ["display_name", "name", "admin_level"],
    )
    place_name = str(result.get("display_name") or result.get("name") or place_query)
    return int(result["id"]), place_name


def resolve_taxon(taxon_query: str) -> tuple[int, str]:
    """Resolve a taxon or family name to an iNaturalist taxon ID."""
    payload = http_get_json("/taxa/autocomplete", {"q": taxon_query, "per_page": 30})
    result = choose_best_result(
        payload.get("results", []),
        taxon_query,
        ["matched_term", "name", "preferred_common_name"],
    )
    taxon_name = str(result.get("name") or taxon_query)
    return int(result["id"]), taxon_name
