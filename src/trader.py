from web3 import Web3
from .config import (
    BASE_RPC, PRIVATE_KEY, WALLET_ADDRESS, BONDING_CURVE_ADDRESS,
    DRY_RUN, MIN_AMOUNT_ETH, MAX_CYCLE_BUY_ETH
)
import logging

w3 = Web3(Web3.HTTPProvider(BASE_RPC))
account = w3.eth.account.from_key(PRIVATE_KEY)

# Minimal ABI – update after finding real contract
BONDING_ABI = [
    {"inputs": [{"name": "amountIn", "type": "uint256"}],
     "name": "buy", "outputs": [], "stateMutability": "payable", "type": "function"},
    # add sell if known
]

bonding = w3.eth.contract(address=BONDING_CURVE_ADDRESS, abi=BONDING_ABI)

def execute_buy(token_address, eth_amount):
    if DRY_RUN:
        logging.info(f"[DRY RUN] Would BUY {eth_amount:.6f} ETH of {token_address}")
        return None

    if eth_amount < MIN_AMOUNT_ETH:
        return None

    amount_wei = w3.to_wei(eth_amount, "ether")
    try:
        tx = bonding.functions.buy(amount_wei).build_transaction({
            "from": WALLET_ADDRESS,
            "value": amount_wei,
            "gas": 400000,
            "gasPrice": w3.eth.gas_price,
            "nonce": w3.eth.get_transaction_count(WALLET_ADDRESS),
        })
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logging.info(f"BUY executed: {token_address} - tx {tx_hash.hex()}")
        return receipt
    except Exception as e:
        logging.error(f"Buy failed for {token_address}: {e}")
        return None

# Add execute_sell similarly when sell ABI is known