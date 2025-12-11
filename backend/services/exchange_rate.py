"""
Exchange Rate Service - USD to CZK conversion.

Uses CNB (Czech National Bank) API for official rates with fallback.
Caches rate for 24 hours to minimize API calls.

Usage:
    from backend.services.exchange_rate import get_usd_to_czk_rate, usd_to_czk

    # Get current exchange rate
    rate = await get_usd_to_czk_rate()

    # Convert USD to CZK
    cost_czk = usd_to_czk(0.05, rate)
"""

import logging
from typing import Optional

import httpx

from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# Fallback rate if CNB API fails (update periodically based on market rates)
FALLBACK_RATE = 23.50

# Sanity check bounds for USD/CZK rate (prevent corrupted data from CNB)
RATE_MIN = 15.0
RATE_MAX = 40.0

# Cache duration: 24 hours (86400 seconds)
CACHE_DURATION_SECONDS = 24 * 3600

# Thread-safe TTL cache for exchange rate (SSOT pattern from src/utils/cache.py)
_rate_cache: TTLCache[float] = TTLCache(
    ttl_seconds=CACHE_DURATION_SECONDS,
    max_size=1,
    name="exchange_rate"
)

# Cache key constant
_CACHE_KEY = "usd_czk_rate"


async def get_usd_to_czk_rate() -> float:
    """
    Get current USD to CZK exchange rate.

    Uses CNB (Czech National Bank) official rates.
    Falls back to hardcoded rate if API fails.
    Caches the rate for 24 hours (thread-safe).

    Returns:
        Exchange rate (CZK per 1 USD)
    """
    # Check cache first (thread-safe TTLCache)
    cached_rate = _rate_cache.get(_CACHE_KEY)
    if cached_rate is not None:
        return cached_rate

    # Fetch from CNB with specific exception handling
    try:
        rate = await _fetch_cnb_rate()
        if rate is not None:
            # Validate rate is within reasonable bounds
            if not (RATE_MIN <= rate <= RATE_MAX):
                logger.error(
                    f"CNB rate {rate:.4f} outside expected range [{RATE_MIN}-{RATE_MAX}], "
                    f"using fallback"
                )
            else:
                # Cache the valid rate
                _rate_cache.set(_CACHE_KEY, rate)
                logger.info(f"Fetched USD/CZK rate from CNB: {rate:.4f}")
                return rate
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching CNB exchange rate: {e}")
    except httpx.TimeoutException as e:
        logger.error(f"Timeout fetching CNB exchange rate: {e}")
    except ValueError as e:
        logger.error(f"Failed to parse CNB exchange rate response: {e}")

    # Use fallback - log at ERROR level since this affects billing accuracy
    logger.error(f"Using fallback USD/CZK rate: {FALLBACK_RATE}")
    return FALLBACK_RATE


async def _fetch_cnb_rate() -> Optional[float]:
    """
    Fetch USD/CZK rate from Czech National Bank API.

    CNB provides daily exchange rates in text format:
    https://www.cnb.cz/en/financial-markets/foreign-exchange-market/
    central-bank-exchange-rate-fixing/central-bank-exchange-rate-fixing/daily.txt

    Format: "Country|Currency|Amount|Code|Rate"
    Example: "USA|dollar|1|USD|23.456"

    Returns:
        Exchange rate (CZK per 1 USD) or None if failed

    Raises:
        httpx.HTTPError: If HTTP request fails
        httpx.TimeoutException: If request times out
        ValueError: If response parsing fails
    """
    url = (
        "https://www.cnb.cz/en/financial-markets/foreign-exchange-market/"
        "central-bank-exchange-rate-fixing/central-bank-exchange-rate-fixing/daily.txt"
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        # Parse CNB format: find USD line
        # Format: "Country|Currency|Amount|Code|Rate"
        for line in response.text.split("\n"):
            if "|USD|" in line:
                parts = line.split("|")
                if len(parts) >= 5:
                    amount = int(parts[2])  # Usually 1
                    if amount <= 0:
                        raise ValueError(f"Invalid amount in CNB response: {amount}")
                    rate = float(parts[4].replace(",", "."))
                    czk_per_usd = rate / amount
                    return czk_per_usd

    logger.warning("USD not found in CNB response")
    return None


def usd_to_czk(usd_amount: float, rate: Optional[float] = None) -> float:
    """
    Convert USD to CZK.

    Args:
        usd_amount: Amount in USD
        rate: Exchange rate (CZK per 1 USD). If None, uses fallback rate.

    Returns:
        Amount in CZK (rounded to 2 decimal places)
    """
    if rate is None:
        rate = FALLBACK_RATE
    return round(usd_amount * rate, 2)


def get_fallback_rate() -> float:
    """Get the fallback exchange rate."""
    return FALLBACK_RATE


def clear_rate_cache() -> None:
    """Clear the cached exchange rate (for testing)."""
    _rate_cache.clear()
    logger.debug("Exchange rate cache cleared")
