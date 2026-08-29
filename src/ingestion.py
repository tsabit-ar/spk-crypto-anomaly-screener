"""Ingestion module for Binance Futures USDT-M public data.

This module provides modular functions to fetch market data from Binance Futures
public endpoints (fapi.binance.com) with $0 cost (no API key required).
Includes built-in resilient DNS-over-HTTPS (DoH) resolution to bypass ISP-level
DNS poisoning/blocks seamlessly.
"""

import os
from typing import Any, Dict, List, Optional
import socket
import requests
from urllib3.util import connection

BINANCE_DEFAULT_BASE_URL = "https://fapi.binance.com"
DEFAULT_TIMEOUT = 10  # seconds

def get_base_url() -> str:
    """Get the Binance Futures Base URL (supports custom proxy via BINANCE_BASE_URL env var)."""
    return os.getenv("BINANCE_BASE_URL", BINANCE_DEFAULT_BASE_URL).rstrip("/")

# In-memory DNS cache for resolved hostnames
_dns_cache: Dict[str, str] = {}

# Known ISP block / DNS poisoning indicators
_POISONED_IPS = {"202.3.218.139", "118.98.117.200", "127.0.0.1", "0.0.0.0"}


def _resolve_doh(host: str) -> Optional[str]:
    """Resolve hostname using DNS-over-HTTPS (Cloudflare / Google) if system DNS is poisoned."""
    if host in _dns_cache:
        return _dns_cache[host]

    # 1. Try Cloudflare DoH (1.1.1.1)
    try:
        url = f"https://1.1.1.1/dns-query?name={host}&type=A"
        headers = {"Accept": "application/dns-json"}
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            answers = resp.json().get("Answer", [])
            for ans in answers:
                if ans.get("type") == 1:  # Type A record
                    ip = ans.get("data")
                    if ip and ip not in _POISONED_IPS:
                        _dns_cache[host] = ip
                        return ip
    except Exception:
        pass

    # 2. Fallback to Google DoH (dns.google)
    try:
        url = f"https://dns.google/resolve?name={host}&type=A"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            answers = resp.json().get("Answer", [])
            for ans in answers:
                if ans.get("type") == 1:
                    ip = ans.get("data")
                    if ip and ip not in _POISONED_IPS:
                        _dns_cache[host] = ip
                        return ip
    except Exception:
        pass

    return None


# Patch urllib3 socket connection to use DoH resolution for binance domains
_original_create_connection = connection.create_connection


def _patched_create_connection(address, *args, **kwargs):
    host, port = address
    if "binance.com" in host:
        try:
            # Check if system DNS resolved to poisoned IP
            sys_ip = socket.gethostbyname(host)
            if sys_ip in _POISONED_IPS:
                resolved_ip = _resolve_doh(host)
                if resolved_ip:
                    return _original_create_connection((resolved_ip, port), *args, **kwargs)
        except Exception:
            resolved_ip = _resolve_doh(host)
            if resolved_ip:
                return _original_create_connection((resolved_ip, port), *args, **kwargs)

        # Proactive DoH resolution for binance domains
        doh_ip = _resolve_doh(host)
        if doh_ip:
            return _original_create_connection((doh_ip, port), *args, **kwargs)

    return _original_create_connection(address, *args, **kwargs)


connection.create_connection = _patched_create_connection

# Standard session with user-agent header
_session = requests.Session()
_session.headers.update({
    "User-Agent": "SPK-CryptoAnomalyScreener/1.0",
    "Accept": "application/json",
})


def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Helper method to send GET requests to Binance Futures API.

    Args:
        endpoint: API path starting with '/' (e.g. '/fapi/v1/ticker/24hr').
        params: Optional dictionary of query parameters.

    Returns:
        JSON parsed response data.

    Raises:
        requests.HTTPError: If the HTTP request returned an unsuccessful status code.
        requests.RequestException: If network or connection errors occur.
    """
    base_url = get_base_url()
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{base_url}{path}"
    response = _session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_bulk_24h_tickers() -> List[Dict[str, Any]]:
    """Fetch 24h rolling window price change statistics for all symbols.

    Endpoint: GET /fapi/v1/ticker/24hr

    Returns:
        List of dictionaries containing 24hr ticker data for all futures symbols.
    """
    data = _get("/fapi/v1/ticker/24hr")
    if isinstance(data, list):
        return data
    return [data]


def fetch_funding_rate(symbol: str) -> Dict[str, Any]:
    """Fetch mark price and funding rate for a specific symbol.

    Endpoint: GET /fapi/v1/premiumIndex

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').

    Returns:
        Dictionary containing markPrice, lastFundingRate, nextFundingTime, etc.
    """
    params = {"symbol": symbol.upper()}
    data = _get("/fapi/v1/premiumIndex", params=params)
    return data


def fetch_delta_oi_4h(symbol: str) -> List[Dict[str, Any]]:
    """Fetch Open Interest historical statistics (hourly interval) for calculating 4H delta.

    Endpoint: GET /futures/data/openInterestHist

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').

    Returns:
        List of Open Interest history records (5 records representing 4 hours delta).
    """
    params = {
        "symbol": symbol.upper(),
        "period": "1h",
        "limit": 5,
    }
    data = _get("/futures/data/openInterestHist", params=params)
    return data


def fetch_klines_1h(symbol: str, limit: int = 20) -> List[List[Any]]:
    """Fetch 1-hour candlestick/kline data for a specific symbol.

    Endpoint: GET /fapi/v1/klines

    Kline format per element:
    [
        0: Open time,
        1: Open price,
        2: High price,
        3: Low price,
        4: Close price,
        5: Volume,
        6: Close time,
        7: Quote asset volume,
        8: Number of trades,
        9: Taker buy base asset volume,
        10: Taker buy quote asset volume,
        11: Ignore
    ]

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').
        limit: Number of candles to retrieve (default: 20, max: 1500).

    Returns:
        List of klines data arrays.
    """
    params = {
        "symbol": symbol.upper(),
        "interval": "1h",
        "limit": limit,
    }
    data = _get("/fapi/v1/klines", params=params)
    return data


def fetch_depth_2pct(symbol: str) -> Dict[str, Any]:
    """Fetch orderbook market depth (limit 100) for calculating liquidity & depth.

    Endpoint: GET /fapi/v1/depth

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').

    Returns:
        Dictionary containing 'lastUpdateId', 'bids' [[price, qty], ...],
        and 'asks' [[price, qty], ...].
    """
    params = {
        "symbol": symbol.upper(),
        "limit": 100,
    }
    data = _get("/fapi/v1/depth", params=params)
    return data
