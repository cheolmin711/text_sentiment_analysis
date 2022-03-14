import pandas as pd
import requests

def stock_prices(ticker):
    """
    ticker is the abbreviated symbol for a stock e.g.AAPL
    this function returns the daily price history of the requested stock as a dataframe
    """
    stock_endpoint = 'https://financialmodelingprep.com/api/v3/historical-price-full/'
    response = requests.get(stock_endpoint + ticker + '?apikey=70407133ea11d7284c70bbca4eee2547').json()
    type(response) == dict
    return pd.DataFrame(response['historical'])

def ny_times_articles(keyword):
    url_list = []
    for i in range(50):
        response = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?q=apple&fq=news_desk:Business&page="+str(i)+"&api-key=fO0tDSRQQdU68GkuXbMjt1uA2FYImzVp").json()
        docs = response['response']['docs']
        for item in docs:
            url_list.append(item['web_url'])
    return url_list

while __name__ == "__main__":
    print(stock_prices('AAPL'))