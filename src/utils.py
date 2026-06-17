import numpy as np

def calculate_forward_alpha(stock_future_ret, spy_future_ret):
    """Wylicza nadwyżkową stopę zwrotu (Alfa)"""
    if stock_future_ret is None or spy_future_ret is None:
        raise ValueError("Dane wejściowe nie mogą być None")
    return stock_future_ret - spy_future_ret

def check_data_leakage(train_dates, test_date):
    """Upewnia się, że żadna data treningowa nie jest równa lub późniejsza niż data testowa"""
    for d in train_dates:
        if d >= test_date:
            return True # Wykryto wyciek danych!
    return False