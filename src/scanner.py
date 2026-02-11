import requests
from bs4 import BeautifulSoup
import logging
from .config import ROBINPUMP_URL, HEADERS

logging.basicConfig(level=logging.INFO)

def scan_robinpump_tokens(max_tokens=50, min_mc_usd=2500, min_change_pct=5):
    """
    Scrape 'Live coins' from robinpump.fun homepage.
    Returns list of dicts with ticker, name, mc, change.
    Note: Contract addresses not available → need external mapping.
    """
    try:
        resp = requests.get(ROBINPUMP_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # IMPORTANT: These selectors are approximate — inspect the real page source
        # and update classes/attributes accordingly (they change frequently)
        coin_items = soup.select("div[class*='coin'], div.card, div.item")[:max_tokens]

        candidates = []
        for item in coin_items:
            try:
                ticker = item.select_one("[class*='ticker'], .symbol, span.upper").text.strip().upper()
                name = item.select_one("[class*='name'], h3, .title").text.strip()
                mc_text = item.select_one("[class*='mc'], .market-cap, [data-value]").text.strip()
                change_text = item.select_one("[class*='change'], .pct, [data-change]").text.strip()

                mc_value = float(mc_text.replace("$", "").replace("K", "000").replace(",", "")) if mc_text else 0
                change = float(change_text.replace("%", "").replace("+", "")) if change_text else 0

                if mc_value >= min_mc_usd and change >= min_change_pct:
                    candidates.append({
                        "ticker": ticker,
                        "name": name,
                        "mc_usd": mc_value,
                        "change_pct": change,
                        "address": None  # Placeholder - must resolve later
                    })
            except:
                continue

        logging.info(f"Found {len(candidates)} promising tokens")
        return candidates

    except Exception as e:
        logging.error(f"Scan failed: {e}")
        return []