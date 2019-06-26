# deep reinforcement learning in trading

## Principle

### States

Method1:
[
[p(t),p(t+1),...p[t+m]]
[vol(t),vol(t+1),...vol[t+m]]
]
p(t) = (price(t+n) - price(t))/price(t)
vol(t) = (turnover(t+n) - turnover(t))/turnover(t)

Method2:
[features]

features include CCI, RSI, VR, ATR, WVAD, LON, ZJTJ, Turnover, Uplift etc.

### Action

- Buy, when agent have stock and it can hold instead of buy.
- Sell, when agent not have stock it use no action instead of sell.
note: Buy and Sell all is trading, which need fee. so we define `inaction`:
```
if state == "holding":
    inaction = buy
else:
    inaction = sell
```

### Reward
without Warehouse management:
```
when action is buy:
Returns = price(t+1) - price(t) - price(t) * taxes
when action is sell:
Returns = - price(t) * taxes
when action is hold stock and is inactions:
Returns = price(t+1) - price(t)
when action is not hold stock and is inactions:
Returns = 0
```
note: the taxes default 0.2%