# src/agent.py
"""
AI Agent module for single-token analysis.
Uses LSTM to learn price patterns from historical data and makes trade decisions.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from pycoingecko import CoinGeckoAPI

from .config import (
    PLATFORM_ID,
    DAYS_HISTORY,
    PREDICT_DAYS,
    MIN_DATA_POINTS,
    RISK_TOLERANCE,
    PORTFOLIO_VALUE_USD,
)

# Initialize CoinGecko client (API key is optional)
cg = CoinGeckoAPI()  # add api_key=... if you have one

class LiquidityAnalyzer:
    """Simple liquidity metrics from historical data"""
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def analyze_liquidity(self):
        """
        Returns:
            tuple: (liquidity_ratio, avg_volume, avg_market_cap)
        """
        if self.data is None or self.data.empty:
            return 0.0, 0.0, 0.0

        avg_volume = self.data['volume'].mean()
        avg_mcap = self.data['market_cap'].mean()
        liquidity_ratio = avg_volume / avg_mcap if avg_mcap > 0 else 0.0

        return liquidity_ratio, avg_volume, avg_mcap


class PatternLearner(nn.Module):
    """Simple LSTM model for price time series prediction"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(PatternLearner, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the last time step output
        return self.fc(out[:, -1, :])


class SingleTokenAgent:
    """
    AI agent responsible for analyzing one token:
    - Fetch historical data from CoinGecko
    - Train LSTM on price history
    - Predict short-term future prices
    - Decide buy/sell/hold + position size
    """

    def __init__(self, token_address: str):
        self.token_address = token_address.lower()
        self.data: pd.DataFrame | None = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = PatternLearner()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def fetch_historical_data(self) -> bool:
        """
        Fetch OHLCV-like data (price, volume, market cap) from CoinGecko
        using contract address on Base chain.
        Returns True if successful.
        """
        try:
            data = cg.get_coin_market_chart_from_contract_address(
                id=PLATFORM_ID,
                contract_address=self.token_address,
                vs_currency='usd',
                days=DAYS_HISTORY
            )

            if not data or 'prices' not in data or not data['prices']:
                logging.warning(f"No historical data available for {self.token_address}")
                return False

            # Build DataFrame
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])

            df = prices.merge(volumes, on='timestamp').merge(market_caps, on='timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            self.data = df
            logging.info(f"Fetched {len(self.data)} data points for {self.token_address}")
            return True

        except Exception as e:
            logging.error(f"Failed to fetch data for {self.token_address}: {e}")
            return False

    def analyze_patterns(self) -> bool:
        """
        Train LSTM model on historical prices.
        Returns True if training was possible.
        """
        if self.data is None or len(self.data) < MIN_DATA_POINTS:
            logging.warning(f"Insufficient data points ({len(self.data or [])}) for {self.token_address}")
            return False

        # Prepare price series
        prices = self.data['price'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences
        seq_length = min(5, len(scaled_prices) - 1)  # short sequences for new tokens
        X, y = [], []

        for i in range(len(scaled_prices) - seq_length):
            X.append(scaled_prices[i:i + seq_length])
            y.append(scaled_prices[i + seq_length])

        if not X:
            logging.warning(f"No valid sequences for {self.token_address}")
            return False

        X = np.array(X)
        y = np.array(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Training loop
        epochs = 50
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                logging.debug(f"Token {self.token_address} | Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

        logging.info(f"Pattern analysis completed for {self.token_address}")
        return True

    def predict_future(self) -> list[float] | None:
        """Predict next PREDICT_DAYS prices"""
        if self.data is None:
            return None

        seq_length = min(5, len(self.data) - 1)
        if seq_length < 1:
            return None

        last_sequence = self.scaler.transform(
            self.data['price'][-seq_length:].values.reshape(-1, 1)
        )
        last_sequence = torch.tensor(last_sequence.reshape(1, -1, 1), dtype=torch.float32)

        predictions = []
        current_seq = last_sequence.clone()

        for _ in range(PREDICT_DAYS):
            with torch.no_grad():
                pred_scaled = self.model(current_seq)
            pred_price = self.scaler.inverse_transform(pred_scaled.numpy())[0][0]
            predictions.append(float(pred_price))

            # Append prediction to sequence for next step
            current_seq = torch.cat((current_seq[:, 1:, :], pred_scaled.unsqueeze(1)), dim=1)

        return predictions

    def decide_trade(self) -> tuple[str, float]:
        """
        Returns:
            (position: 'Buy'|'Sell'|'Hold', amount_eth: float)
        """
        if self.data is None or self.data.empty:
            return 'Hold', 0.0

        liquidity_ratio, avg_volume, avg_mcap = LiquidityAnalyzer(self.data).analyze_liquidity()

        predictions = self.predict_future()
        if not predictions:
            return 'Hold', 0.0

        current_price = float(self.data['price'][-1])
        predicted_price = predictions[-1]
        predicted_change = (predicted_price - current_price) / current_price

        # Decision rules (tuned for volatile memecoins)
        if predicted_change > 0.10:
            position = 'Buy'
        elif predicted_change < -0.10:
            position = 'Sell'
        else:
            position = 'Hold'

        amount_eth = 0.0
        if position in ('Buy', 'Sell'):
            # Position sizing: risk-based + predicted move strength
            usd_risk = PORTFOLIO_VALUE_USD * RISK_TOLERANCE * abs(predicted_change)
            usd_risk = min(usd_risk, PORTFOLIO_VALUE_USD * 0.05)  # max 5% of portfolio

            # Reduce size if liquidity is poor
            if liquidity_ratio < 0.001:
                usd_risk *= 0.5

            # Convert USD → ETH (using current ETH price from CoinGecko)
            try:
                eth_price_data = cg.get_price('ethereum', 'usd')
                eth_price = eth_price_data['ethereum']['usd']
                amount_eth = usd_risk / eth_price
            except Exception as e:
                logging.warning(f"Failed to get ETH price: {e} → skipping trade")
                amount_eth = 0.0

        logging.info(
            f"{self.token_address} → {position} | "
            f"predicted change: {predicted_change:+.2%} | "
            f"amount: {amount_eth:.6f} ETH | "
            f"liquidity ratio: {liquidity_ratio:.4f}"
        )

        return position,