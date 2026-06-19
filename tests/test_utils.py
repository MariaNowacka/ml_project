import pytest
import pandas as pd
from utils import calculate_forward_alpha, check_data_leakage

def test_calculate_forward_alpha_correctness():
    # Testujemy, czy proste odejmowanie stóp zwrotu działa poprawnie
    stock_ret = 0.05  # +5%
    spy_ret = 0.02    # +2%
    expected_alpha = 0.03
    
    assert calculate_forward_alpha(stock_ret, spy_ret) == pytest.approx(expected_alpha)

def test_calculate_forward_alpha_raises_value_error():
    # Testujemy, czy funkcja prawidłowo wyrzuca błąd przy braku danych
    with pytest.raises(ValueError):
        calculate_forward_alpha(None, 0.02)

def test_check_data_leakage_safe():
    # Przypadek bezpieczny: trening kończy się przed testem
    train_dates = [pd.Timestamp('2014-01-01'), pd.Timestamp('2014-12-31')]
    test_date = pd.Timestamp('2015-01-01')
    
    assert check_data_leakage(train_dates, test_date) == False

def test_check_data_leakage_detected():
    # Przypadek błędu: dane treningowe zawierają datę z przyszłości (wyciek!)
    train_dates = [pd.Timestamp('2014-01-01'), pd.Timestamp('2015-01-02')]
    test_date = pd.Timestamp('2015-01-01')
    
    assert check_data_leakage(train_dates, test_date) == True