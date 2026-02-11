import os
from dotenv import load_dotenv

load_dotenv()

# Core settings
BASE_RPC = os.getenv("BASE_RPC", "https://mainnet.base.org")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "").lower()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
PLATFORM_ID = "base"
DAYS_HISTORY = 30
PREDICT_DAYS = 3
MIN_DATA_POINTS = 10
MAX_WORKERS = 20

RISK_TOLERANCE = 0.02
PORTFOLIO_VALUE_USD = 10000
MIN_AMOUNT_ETH = float(os.getenv("MIN_AMOUNT_ETH", 0.001))
MAX_CYCLE_BUY_ETH = float(os.getenv("MAX_CYCLE_BUY_ETH", 0.1))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

ROBINPUMP_URL = "https://robinpump.fun"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Placeholder - REPLACE WITH REAL RobinPump bonding curve address
BONDING_CURVE_ADDRESS = "0x0000000000000000000000000000000000000000"  # CRITICAL: FIND THIS!