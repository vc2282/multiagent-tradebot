# RobinPump AI Trader

**AI-powered trading bot for RobinPump.fun (Base chain)**

Scans for active tokens → analyzes price patterns & liquidity with LSTM agents → makes automated buy/sell decisions.

**WARNING**  
This is **experimental high-risk software**. Memecoin trading can lead to **complete loss of capital**.  
Use tiny test amounts. Never run with real money without understanding the code.

## Features

- Homepage scraper for discovering active tokens
- Multi-agent LSTM pattern recognition per token
- Liquidity analysis via CoinGecko data
- Automated trading on bonding curve (Base)
- Dry-run mode for safety
- Parallel processing for 50–100+ tokens

## Setup

1. Clone repo
   ```bash
   git clone https://github.com/yourusername/robinpump-ai-trader.git
   cd robinpump-ai-trader
