import os
import finnhub

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# print(finnhub_client.general_news('AAPL', min_id=0))

# print(finnhub_client.news_sentiment('AAPL'))

# print(finnhub_client.covid19())

# print(finnhub_client.technical_indicator(symbol="AAPL", resolution='D', _from=1583098857, to=1584308457, indicator='rsi', indicator_fields={"timeperiod": 3}))

# print(finnhub_client.recommendation_trends('AAPL'))

# print(finnhub_client.price_target('AAPL'))
