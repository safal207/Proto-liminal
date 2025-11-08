#!/usr/bin/env python3
"""
Module: market_data_api.py
Purpose: Unified interface for fetching market data from multiple sources
Part of LIMINAL ProtoConsciousness MVP - Automated Outcome Collection

Supported APIs:
1. Yahoo Finance - stocks, ETFs, indices
2. Binance - cryptocurrency (primary)
3. CoinGecko - cryptocurrency (fallback, no auth required)
4. Tradernet - Russian stocks + crypto (when available)

Philosophy:
"Truth emerges from multiple perspectives. Use redundant data sources
to ensure reality checks are robust and reliable."
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import requests

LOGGER = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo"
    BINANCE = "binance"
    COINGECKO = "coingecko"
    TRADERNET = "tradernet"


class MarketDataAPI:
    """
    Unified interface for market data from multiple sources

    Features:
    - Automatic source selection based on entity type
    - Fallback to alternative sources on failure
    - Rate limiting and retry logic
    - Caching to minimize API calls
    """

    def __init__(
        self,
        cache_dir: str = "data/market_cache",
        cache_ttl_seconds: int = 300,  # 5 minutes
        rate_limit_per_second: float = 2.0
    ):
        """
        Initialize market data API

        Args:
            cache_dir: Directory for caching responses
            cache_ttl_seconds: Cache time-to-live
            rate_limit_per_second: Max requests per second
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = cache_ttl_seconds
        self.rate_limit = rate_limit_per_second
        self.last_request_time = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0
        }

        LOGGER.info("MarketDataAPI initialized")

    def _rate_limit_wait(self, source: DataSource):
        """Enforce rate limiting"""
        if source not in self.last_request_time:
            self.last_request_time[source] = 0

        elapsed = time.time() - self.last_request_time[source]
        min_interval = 1.0 / self.rate_limit

        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            LOGGER.debug(f"Rate limiting {source.value}: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self.last_request_time[source] = time.time()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> Optional[Dict]:
        """Read from cache if not expired"""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            # Check if expired
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age = datetime.now(timezone.utc) - cached_time

            if age.total_seconds() > self.cache_ttl:
                LOGGER.debug(f"Cache expired for {cache_key}")
                return None

            self.stats["cache_hits"] += 1
            return cached['data']

        except Exception as e:
            LOGGER.warning(f"Cache read error for {cache_key}: {e}")
            return None

    def _write_cache(self, cache_key: str, data: Dict):
        """Write data to cache"""
        cache_path = self._get_cache_path(cache_key)

        try:
            cached = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data
            }

            with open(cache_path, 'w') as f:
                json.dump(cached, f)

        except Exception as e:
            LOGGER.warning(f"Cache write error for {cache_key}: {e}")

    def fetch_price_yahoo(self, symbol: str) -> Optional[float]:
        """
        Fetch current price from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., "AAPL", "TSLA")

        Returns:
            Current price or None
        """
        cache_key = f"yahoo_{symbol}_{int(time.time() // 60)}"

        # Check cache
        cached = self._read_cache(cache_key)
        if cached and 'price' in cached:
            return cached['price']

        # Rate limit
        self._rate_limit_wait(DataSource.YAHOO_FINANCE)

        try:
            # Yahoo Finance API (unofficial)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "interval": "1m",
                "range": "1d"
            }

            self.stats["api_calls"] += 1
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract current price
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice')

            if price:
                # Cache result
                self._write_cache(cache_key, {'price': price})
                LOGGER.debug(f"Yahoo Finance: {symbol} = ${price:.2f}")
                return float(price)

        except Exception as e:
            LOGGER.error(f"Yahoo Finance error for {symbol}: {e}")
            self.stats["errors"] += 1

        return None

    def fetch_price_binance(self, symbol: str) -> Optional[float]:
        """
        Fetch current price from Binance

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")

        Returns:
            Current price or None
        """
        cache_key = f"binance_{symbol}_{int(time.time() // 60)}"

        # Check cache
        cached = self._read_cache(cache_key)
        if cached and 'price' in cached:
            return cached['price']

        # Rate limit
        self._rate_limit_wait(DataSource.BINANCE)

        try:
            # Binance public API
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": symbol}

            self.stats["api_calls"] += 1
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = data.get('price')

            if price:
                price = float(price)
                # Cache result
                self._write_cache(cache_key, {'price': price})
                LOGGER.debug(f"Binance: {symbol} = ${price:.2f}")
                return price

        except Exception as e:
            LOGGER.error(f"Binance error for {symbol}: {e}")
            self.stats["errors"] += 1

        return None

    def fetch_price_coingecko(self, coin_id: str, vs_currency: str = "usd") -> Optional[float]:
        """
        Fetch current price from CoinGecko (no auth required)

        Args:
            coin_id: CoinGecko coin ID (e.g., "bitcoin", "ethereum")
            vs_currency: Quote currency (default: "usd")

        Returns:
            Current price or None
        """
        cache_key = f"coingecko_{coin_id}_{vs_currency}_{int(time.time() // 60)}"

        # Check cache
        cached = self._read_cache(cache_key)
        if cached and 'price' in cached:
            return cached['price']

        # Rate limit
        self._rate_limit_wait(DataSource.COINGECKO)

        try:
            # CoinGecko public API
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": vs_currency
            }

            self.stats["api_calls"] += 1
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = data.get(coin_id, {}).get(vs_currency)

            if price:
                price = float(price)
                # Cache result
                self._write_cache(cache_key, {'price': price})
                LOGGER.debug(f"CoinGecko: {coin_id} = ${price:.2f}")
                return price

        except Exception as e:
            LOGGER.error(f"CoinGecko error for {coin_id}: {e}")
            self.stats["errors"] += 1

        return None

    def normalize_symbol(self, entity: str) -> Tuple[str, DataSource]:
        """
        Normalize entity name to API-specific symbol

        Args:
            entity: Entity name (e.g., "AAPL", "BTC/USD", "Bitcoin")

        Returns:
            (normalized_symbol, preferred_source)
        """
        entity_upper = entity.upper()

        # Crypto pairs
        if "/" in entity:
            # Format: BTC/USD, ETH/USDT
            base, quote = entity.split("/")
            base = base.replace("BTC", "BTC").replace("ETH", "ETH")
            quote = quote.replace("USD", "USDT").replace("USDT", "USDT")

            # Binance format: BTCUSDT
            binance_symbol = f"{base}{quote}"
            return (binance_symbol, DataSource.BINANCE)

        # Known crypto symbols
        crypto_symbols = {
            "BITCOIN": "BTCUSDT",
            "BTC": "BTCUSDT",
            "ETHEREUM": "ETHUSDT",
            "ETH": "ETHUSDT",
            "BTCUSD": "BTCUSDT",
            "ETHUSD": "ETHUSDT"
        }

        if entity_upper in crypto_symbols:
            return (crypto_symbols[entity_upper], DataSource.BINANCE)

        # Default to stocks (Yahoo Finance)
        return (entity, DataSource.YAHOO_FINANCE)

    def fetch_price(self, entity: str, source: Optional[DataSource] = None) -> Optional[float]:
        """
        Fetch current price with automatic source selection

        Args:
            entity: Entity name or symbol
            source: Optional explicit source (auto-detect if None)

        Returns:
            Current price or None
        """
        self.stats["total_requests"] += 1

        # Normalize symbol and detect source
        if source is None:
            normalized_symbol, preferred_source = self.normalize_symbol(entity)
        else:
            normalized_symbol = entity
            preferred_source = source

        # Try preferred source
        if preferred_source == DataSource.YAHOO_FINANCE:
            price = self.fetch_price_yahoo(normalized_symbol)
            if price:
                return price

        elif preferred_source == DataSource.BINANCE:
            price = self.fetch_price_binance(normalized_symbol)
            if price:
                return price

            # Fallback to CoinGecko for crypto
            coin_id_map = {
                "BTCUSDT": "bitcoin",
                "ETHUSDT": "ethereum"
            }
            if normalized_symbol in coin_id_map:
                LOGGER.debug(f"Binance failed, trying CoinGecko for {entity}")
                price = self.fetch_price_coingecko(coin_id_map[normalized_symbol])
                if price:
                    return price

        return None

    def get_stats(self) -> Dict:
        """Get API statistics"""
        cache_hit_rate = (
            self.stats["cache_hits"] / max(1, self.stats["total_requests"])
        )

        return {
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "api_calls": self.stats["api_calls"],
            "cache_hit_rate": cache_hit_rate,
            "errors": self.stats["errors"],
            "error_rate": self.stats["errors"] / max(1, self.stats["api_calls"])
        }


def main():
    """CLI interface for testing market data API"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument("symbols", nargs="+", help="Symbols to fetch (e.g., AAPL BTC/USD)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create API
    api = MarketDataAPI()

    print("\n" + "="*60)
    print("Market Data Fetcher")
    print("="*60)

    # Fetch prices
    for symbol in args.symbols:
        print(f"\n{symbol}:")
        price = api.fetch_price(symbol)

        if price:
            print(f"  Price: ${price:.2f}")
        else:
            print(f"  ❌ Failed to fetch price")

    # Print stats
    stats = api.get_stats()
    print("\n" + "="*60)
    print("Statistics")
    print("="*60)
    print(f"Total requests:   {stats['total_requests']}")
    print(f"Cache hits:       {stats['cache_hits']}")
    print(f"API calls:        {stats['api_calls']}")
    print(f"Cache hit rate:   {stats['cache_hit_rate']:.2%}")
    print(f"Errors:           {stats['errors']}")
    print(f"Error rate:       {stats['error_rate']:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()
