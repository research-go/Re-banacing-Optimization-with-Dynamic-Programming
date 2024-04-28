import numpy as np


def sharpe_ratio(sequence, data_series_period, risk_free=0.0):
    """
    sharpe ratio
    :param sequence: data monthly price(not return) sequence
    :param data_series_period: how long(years)
    :param risk_free: risk_free
    :return: float
    """
    daily_return = []
    before = sequence[0]
    for i in sequence[1:]:
        daily_return.append(i/before - 1)
        before = i
    pf_return = (sequence[-1]/sequence[0])**(1/data_series_period) - 1
    std = np.std(daily_return)*(12**(1/2))  # monthly data standard deviation to the annual
    ratio = float((pf_return - risk_free)/std)
    return ratio


if __name__ == "__main__":
    return_list = [1 + 0.1*i for i in range(12)]
    print(sharpe_ratio(return_list, 10))
    print(sharpe_ratio(return_list, 10))
