import time
from src.scanner import scan_robinpump_tokens
from src.agent import SingleTokenAgent
from src.trader import execute_buy
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import MAX_WORKERS, PORTFOLIO_VALUE_USD, RISK_TOLERANCE
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class RobinPumpAIBot:
    def __init__(self):
        self.known_addresses = set()  # Add known ones manually or from past scans

    def refresh_candidates(self):
        candidates = scan_robinpump_tokens(max_tokens=80)
        new_addrs = []
        for c in candidates:
            # Critical: resolve real address (placeholder logic)
            # In production: use Basescan API / event log / manual list
            addr = f"0xplaceholder_{c['ticker']}"  # ← REPLACE THIS LOGIC
            if addr not in self.known_addresses:
                self.known_addresses.add(addr)
                new_addrs.append(addr)
        logging.info(f"Active tokens: {len(self.known_addresses)}")
        return list(self.known_addresses)

    def run_cycle(self):
        addrs = self.refresh_candidates()
        agents = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(SingleTokenAgent(addr).analyze_patterns): addr 
                       for addr in addrs if SingleTokenAgent(addr).fetch_historical_data()}
            for future in as_completed(futures):
                addr = futures[future]
                try:
                    if future.result():  # if trained
                        agents[addr] = SingleTokenAgent(addr)
                except Exception as e:
                    logging.debug(f"{addr} skipped: {e}")

        total_buy = 0.0
        for addr, agent in agents.items():
            pos, amt_eth = agent.decide_trade()
            if pos == "Buy" and amt_eth >= 0.001:
                if total_buy + amt_eth > MAX_CYCLE_BUY_ETH:
                    logging.warning(f"Buy cap reached – skipping {addr}")
                    continue
                execute_buy(addr, amt_eth)
                total_buy += amt_eth

        logging.info(f"Cycle finished | Buys: {total_buy:.4f} ETH")

    def run(self, interval=900):  # 15 min
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logging.error(f"Cycle error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    bot = RobinPumpAIBot()
    logging.info("RobinPump AI Trader started")
    bot.run()