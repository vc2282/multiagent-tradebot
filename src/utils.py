# src/utils.py
"""
Utility functions for the RobinPump AI Trader bot.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from pycoingecko import CoinGeckoAPI
from web3 import Web3

from .config import BASE_RPC


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging with console and optional file output.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        ))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=handlers,
        force=True
    )

    # Suppress noisy logs from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("web3").setLevel(logging.WARNING)


def get_current_eth_price() -> Optional[float]:
    """
    Fetch current ETH price in USD from CoinGecko.
    Returns None if the request fails.
    """
    try:
        cg = CoinGeckoAPI()
        price_data = cg.get_price('ethereum', 'usd')
        if price_data and 'ethereum' in price_data and 'usd' in price_data['ethereum']:
            return float(price_data['ethereum']['usd'])
        else:
            logging.warning("Could not retrieve ETH price from CoinGecko")
            return None
    except Exception as e:
        logging.error(f"Failed to get ETH price: {e}")
        return None


def wei_to_ether(wei_amount: int) -> float:
    """Convert wei to ETH (human-readable)"""
    return Web3.from_wei(wei_amount, 'ether')


def ether_to_wei(ether_amount: float) -> int:
    """Convert ETH to wei"""
    return Web3.to_wei(ether_amount, 'ether')


def get_timestamp_str() -> str:
    """Return current timestamp as string (useful for logging/filenames)"""
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def format_float(value: float, decimals: int = 6) -> str:
    """Format float with fixed decimal places, avoiding scientific notation"""
    return f"{value:.{decimals}f}"


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, return default on failure"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default