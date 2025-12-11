"""
Exchange Rate Service - USD to CZK conversion.

Uses CNB (Czech National Bank) API for official rates with fallback.
Caches rate for 24 hours to minimize API calls.

Usage:
    from backend.services.exchange_rate import get_usd_to_czk_rate, usd_to_czk

    # Get current exchange rate
    rate = await get_usd_to_czk_rate()

    # Convert USD to CZK
    cost_czk = usd_to_czk(0.05, rate)  # ~1.18 CZK
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Fallback rate if CNB API fails (approximate CZK/USD as of Dec 2024)
FALLBACK_RATE = 23.50

# Cache: (rate, expiration_time)
_rate_cache: Tuple[Optional[float], Optional[datetime]] = (None, None)

# Cache duration: 24 hours
CACHE_DURATION_HOURS = 24


async def get_usd_to_czk_rate() -> float:
    """
    Get current USD to CZK exchange rate.

    Uses CNB (Czech National Bank) official rates.
    Falls back to hardcoded rate if API fails.
    Caches the rate for 24 hours.

    Returns:
        Exchange rate (CZK per 1 USD)
    """
    global _rate_cache

    # Check cache first
    cached_rate, expiration = _rate_cache
    if cached_rate is not None and expiration is not None:
        if datetime.now() < expiration:
            return cached_rate

    # Fetch from CNB
    try:
        rate = await _fetch_cnb_rate()
        if rate is not None:
            # Cache the rate for 24 hours
            _rate_cache = (rate, datetime.now() + timedelta(hours=CACHE_DURATION_HOURS))
            logger.info(f"Fetched USD/CZK rate from CNB: {rate:.4f}")
            return rate
    except Exception as e:
        logger.warning(f"Failed to fetch CNB exchange rate: {e}")

    # Use fallback
    logger.warning(f"Using fallback USD/CZK rate: {FALLBACK_RATE}")
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
    global _rate_cache
    _rate_cache = (None, None)
    logger.debug("Exchange rate cache cleared")
