# 🔍 **TRADING CALCULATIONS EXPLAINED**

## **How the Financial AI System Determines Short/Long and Price Points**

This document explains the **exact mathematical calculations** used by the system to determine trading decisions and price levels.

---

## 📊 **1. OVERALL SETUP SCORE CALCULATION**

The system calculates a **composite setup score** (0-1) using this weighted formula:

```
Setup Score = (Trend Score × 0.30) + 
              (Momentum Score × 0.25) + 
              (Volatility Score × 0.20) + 
              (Volume Score × 0.15) + 
              (Support/Resistance Score × 0.10)
```

### **Score Thresholds:**
- **≥ 0.8**: Strong Setup (Strong Buy/Sell)
- **≥ 0.6**: Good Setup (Buy/Sell Setup)  
- **≥ 0.4**: Weak Setup
- **< 0.4**: No Setup

---

## 🎯 **2. TREND SCORE CALCULATION (30% Weight)**

### **Formula:**
```
Trend Score = (MA Trend × 0.40) + (Price vs MA × 0.30) + (MA Slope × 0.30)
```

### **Components:**

#### **A) MA Trend (40%):**
```python
if SMA_20 > SMA_50:
    MA_Trend = 1.0  # Bullish
else:
    MA_Trend = 0.0  # Bearish
```

#### **B) Price vs MA (30%):**
```python
if Current_Price > SMA_20:
    Price_vs_MA = 1.0  # Price above MA
else:
    Price_vs_MA = 0.0  # Price below MA
```

#### **C) MA Slope (30%):**
```python
MA_Slope = (SMA_20[-1] - SMA_20[-10]) / SMA_20[-10]
Slope_Score = min(max(MA_Slope × 10, 0), 1)  # Normalize to 0-1
```

---

## 🚀 **3. MOMENTUM SCORE CALCULATION (25% Weight)**

### **Base Score: 0.5**

### **RSI Analysis:**
```python
if RSI < 30:          # Oversold
    Momentum_Score += 0.3
elif RSI > 70:        # Overbought  
    Momentum_Score -= 0.3
elif 40 ≤ RSI ≤ 60:   # Neutral
    Momentum_Score += 0.1
```

### **MACD Analysis:**
```python
if MACD > MACD_Signal and MACD_Histogram > 0:
    Momentum_Score += 0.2    # Bullish momentum
elif MACD < MACD_Signal and MACD_Histogram < 0:
    Momentum_Score -= 0.2    # Bearish momentum
```

### **Price Momentum:**
```python
Price_Change = (Close[-1] - Close[-5]) / Close[-5]
if abs(Price_Change) > 0.02:  # 2% change
    if Price_Change > 0:
        Momentum_Score += 0.1  # Bullish
    else:
        Momentum_Score -= 0.1  # Bearish
```

---

## 📈 **4. VOLATILITY SCORE CALCULATION (20% Weight)**

### **Formula:**
```python
Vol_Ratio = Current_Volatility / Average_Volatility
Volatility_Score = min(Vol_Ratio, 2.0) / 2.0  # Normalize to 0-1
```

### **Example:**
- Current Volatility: 0.03 (3%)
- Average Volatility: 0.02 (2%)
- Vol Ratio: 0.03 / 0.02 = 1.5
- Volatility Score: 1.5 / 2.0 = 0.75

**Higher volatility = Better for CFD trading**

---

## 📊 **5. VOLUME SCORE CALCULATION (15% Weight)**

### **Volume Ratio Scoring:**
```python
Volume_Ratio = Current_Volume / Average_Volume(20_periods)

if Volume_Ratio > 1.5:
    Volume_Score = 1.0      # High volume
elif Volume_Ratio > 1.0:
    Volume_Score = 0.7      # Above average
elif Volume_Ratio > 0.5:
    Volume_Score = 0.4      # Below average
else:
    Volume_Score = 0.2      # Low volume
```

---

## 🎯 **6. SUPPORT/RESISTANCE SCORE (10% Weight)**

### **Formula:**
```python
Proximity_Score = 1.0 - (Min_Distance / Price_Range)
```

### **Example:**
- Current Price: $100
- Nearest Support: $98 (2 points away)
- Nearest Resistance: $102 (2 points away)
- Price Range: $102 - $98 = $4
- Min Distance: $2
- Proximity Score: 1.0 - (2/4) = 0.5

**Closer to levels = Higher score**

---

## 🔄 **7. DIRECTION DETERMINATION**

### **Decision Logic:**
```python
if Setup_Score >= 0.6:
    if Trend_Direction == 'bullish':
        Direction = 'long'
        Setup_Type = 'Buy Setup' or 'Strong Buy'
    elif Trend_Direction == 'bearish':
        Direction = 'short'
        Setup_Type = 'Sell Setup' or 'Strong Sell'
    else:
        Direction = 'neutral'
else:
    Direction = 'neutral'
```

---

## 💰 **8. ENTRY PRICE CALCULATION**

### **For LONG Positions:**
```python
Entry_Levels = {
    'aggressive': Current_Price × 0.995,    # 0.5% below current
    'moderate': Current_Price × 0.99,       # 1.0% below current  
    'conservative': Current_Price × 0.985   # 1.5% below current
}
```

### **For SHORT Positions:**
```python
Entry_Levels = {
    'aggressive': Current_Price × 1.005,    # 0.5% above current
    'moderate': Current_Price × 1.01,       # 1.0% above current
    'conservative': Current_Price × 1.015   # 1.5% above current
}
```

### **Example - LONG AAPL:**
- Current Price: $150.00
- Aggressive Entry: $150.00 × 0.995 = **$149.25**
- Moderate Entry: $150.00 × 0.99 = **$148.50**
- Conservative Entry: $150.00 × 0.985 = **$147.75**

---

## 🛑 **9. STOP LOSS CALCULATION**

### **For LONG Positions:**
```python
Stop_Loss = Entry_Price × 0.97  # 3% below entry
```

### **For SHORT Positions:**
```python
Stop_Loss = Entry_Price × 1.03  # 3% above entry
```

### **Example - LONG AAPL:**
- Moderate Entry: $148.50
- Stop Loss: $148.50 × 0.97 = **$144.05**
- Risk: $148.50 - $144.05 = **$4.45 per share**

---

## 🎯 **10. TAKE PROFIT CALCULATION**

### **Formula (2:1 Risk/Reward):**
```python
Risk = |Entry_Price - Stop_Loss|

# For LONG:
Take_Profit = Entry_Price + (Risk × 2)

# For SHORT:
Take_Profit = Entry_Price - (Risk × 2)
```

### **Example - LONG AAPL:**
- Entry: $148.50
- Stop Loss: $144.05
- Risk: $4.45
- Take Profit: $148.50 + ($4.45 × 2) = **$157.40**

**Risk/Reward Ratio: $4.45 risk vs $8.90 reward = 2:1**

---

## 📊 **11. COMPLETE CALCULATION EXAMPLE**

### **Scenario: AAPL Analysis**

#### **Step 1: Calculate Individual Scores**
```python
Trend_Score = 0.85      # Strong bullish trend
Momentum_Score = 0.75   # Good momentum
Volatility_Score = 0.80 # High volatility
Volume_Score = 0.90     # High volume
Support_Resistance_Score = 0.70  # Near levels
```

#### **Step 2: Calculate Setup Score**
```python
Setup_Score = (0.85 × 0.30) + (0.75 × 0.25) + (0.80 × 0.20) + 
              (0.90 × 0.15) + (0.70 × 0.10)

Setup_Score = 0.255 + 0.1875 + 0.16 + 0.135 + 0.07
Setup_Score = 0.8075  # Strong Setup (≥ 0.8)
```

#### **Step 3: Determine Direction**
```python
if Setup_Score >= 0.8 and Trend_Direction == 'bullish':
    Direction = 'long'
    Setup_Type = 'Strong Buy'
    Confidence = 0.8075
```

#### **Step 4: Calculate Price Levels**
```python
Current_Price = $150.00

# Entry Levels (LONG)
Aggressive_Entry = $150.00 × 0.995 = $149.25
Moderate_Entry = $150.00 × 0.99 = $148.50
Conservative_Entry = $150.00 × 0.985 = $147.75

# Stop Loss
Stop_Loss = $148.50 × 0.97 = $144.05

# Take Profit
Risk = $148.50 - $144.05 = $4.45
Take_Profit = $148.50 + ($4.45 × 2) = $157.40
```

---

## 🎯 **12. KEY DECISION FACTORS**

### **Primary Factors (70% combined weight):**
1. **Trend Analysis (30%)** - Moving average alignment and slope
2. **Momentum (25%)** - RSI, MACD, price momentum
3. **Volatility (20%)** - Current vs average volatility

### **Secondary Factors (30% combined weight):**
4. **Volume (15%)** - Current vs average volume
5. **Support/Resistance (10%)** - Proximity to key levels

### **Direction Decision:**
- **LONG**: Setup Score ≥ 0.6 + Bullish Trend
- **SHORT**: Setup Score ≥ 0.6 + Bearish Trend
- **NEUTRAL**: Setup Score < 0.6 or Mixed Signals

---

## 🔧 **13. SYSTEM CONFIGURATION**

### **Current Settings:**
```python
setup_criteria = {
    'trend_strength': 0.7,        # Minimum trend strength
    'volatility_threshold': 0.02,  # Minimum volatility (2%)
    'volume_ratio': 1.5,          # Minimum volume ratio
    'risk_reward_ratio': 2.0,     # Minimum risk/reward
    'setup_confidence': 0.6       # Minimum confidence
}
```

### **Risk Management:**
- **Position Size**: 2% of account per trade
- **Stop Loss**: 3% from entry
- **Take Profit**: 2:1 risk/reward ratio
- **Max Risk**: £2.94 per trade (£147 × 2%)

---

## 📈 **14. REAL-TIME UPDATES**

The system recalculates **every 30 seconds** using:
- **Live price data** from Yahoo Finance
- **Real-time technical indicators**
- **Current market conditions**
- **Live news sentiment**

### **Update Frequency:**
- **Price Data**: Every 30 seconds
- **Technical Indicators**: Every 1 minute
- **Market Status**: Every 1 minute
- **News Sentiment**: Every 5 minutes

---

## 🎯 **15. SUMMARY**

### **Short/Long Decision Process:**
1. **Calculate 5 component scores** (Trend, Momentum, Volatility, Volume, S/R)
2. **Weight and combine** into overall setup score
3. **Apply threshold** (≥ 0.6 for trading setup)
4. **Check trend direction** (bullish = long, bearish = short)
5. **Generate confidence level** (0.6 to 1.0)

### **Price Point Calculation:**
1. **Entry**: Current price ± 0.5-1.5% (pullback for long, bounce for short)
2. **Stop Loss**: Entry ± 3% (below for long, above for short)
3. **Take Profit**: 2:1 risk/reward ratio from entry

### **Example Output:**
```
AAPL: Strong Buy (LONG)
- Entry: $148.50 (moderate)
- Stop Loss: $144.05 (3% risk)
- Take Profit: $157.40 (2:1 reward)
- Confidence: 80.75%
- Setup Score: 0.8075
```

This system provides **mathematically precise, risk-managed trading decisions** based on real-time market data and professional technical analysis principles.