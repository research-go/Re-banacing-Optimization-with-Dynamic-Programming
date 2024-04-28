"""
Written by Byeongjae Kwon
Please put code and csv files together in one directory.
"""

# Libraries
import pandas as pd
import numpy as np
import datetime as dt
import cost
import valuation

# Environment Variables
risk_aversion = 1.0
increment = 0.01  # weight accuracy
one_transaction_cost = 0.002
dc_factor = np.exp(-0.02/12)  # risk-free rate
dp_tol = 0.0001  # dynamic programming convergence tolerance
# there are KOSPI200 TR, KRX BOND TR monthly data from "2011-02-28" to "2024-03-29"
data = pd.read_csv('./data_monthly.csv')


# Expected Quadratic Utility
def utility(mean: float, variance: float, risk_averse: float, method='quadratic') -> float:
    """
    Expected Quadratic Utility. you can substitute log for the quadratic.
    :param mean: expected return
    :param variance: expected variance
    :param risk_averse: risk aversion factor
    :param method: what function for expected utility
    :return: expected utility
    :rtype: float
    """
    if method == 'quadratic':
        value = mean - (risk_averse/2)*variance
    elif method == 'log':
        value = np.log(1 + mean) - variance/(2*((1 + mean)**2))
    else:
        raise Exception("Expected Utility didn't well defined.")
    return value


class Portfolio:
    def __init__(self, start: dt.datetime, end: dt.datetime, num: int, simulation_years=10):
        """
        we use historical average return, variance for expected return, variance(covariance matrix)
        :param start: period start for average return, variance
        :param end: period end for average return, variance
        :param num: the number of simulation
        """
        self.period_start = start
        self.period_end = end
        self.file = data
        # log return
        self.file['Return_Equity'] = np.log(self.file['Equity']/self.file['Equity'].shift(1))
        self.file['Return_Fixed Income'] = np.log(self.file['Fixed Income']/self.file['Fixed Income'].shift(1))
        self.mean = np.array([np.mean(self.file['Return_Equity'][1:]), np.mean(self.file['Return_Fixed Income'][1:])])
        self.covariance_mat = np.cov(self.file['Return_Equity'][1:], self.file['Return_Fixed Income'][1:])
        # 10-years monthly data sampling. multivariate-gaussian
        self.sample = [np.random.multivariate_normal(self.mean, self.covariance_mat, 12*simulation_years) for x in range(num)]
        # Optimal Weight. Portfolio starts with this
        self.optimal_weight = [0.0, 1.0]
        current_weight = self.optimal_weight
        self.optimal_mean = sum(self.mean*current_weight)
        self.optimal_variance = float(np.dot(np.dot(current_weight, self.covariance_mat), current_weight))
        max_util = utility(self.optimal_mean, self.optimal_variance, risk_aversion)
        for i in range(round(1/increment) + 1):  # Optimal Weight is the argmax utility
            current_weight = [round(increment*i, round(np.log10(1/increment))),
                              round(1 - increment*i, round(np.log10(1/increment)))]
            mean = sum(self.mean*current_weight)
            variance = float(np.dot(np.dot(current_weight, self.covariance_mat), current_weight))
            util = utility(mean, variance, risk_aversion)
            if util > max_util:  # Optimal weight is argmax (util)
                max_util = util
                self.optimal_weight = current_weight
        self.optimal_mean = sum(self.mean*self.optimal_weight)
        self.optimal_variance = float(np.dot(np.dot(self.optimal_weight, self.covariance_mat), self.optimal_weight))
        self.utility = utility(self.optimal_mean, self.optimal_variance, risk_aversion)
        self.policy_basket = []  # for dynamic programming output

    def a_scenario_run(self, asset_returns_list: np.ndarray,
                       strategy='buy_and_hold', period_variable=None,
                       threshold_variable=None, dp_variable=None) -> dict:
        """
        :param asset_returns_list: a scenario returns
        :param strategy: buy_and_hold, periodic, threshold and dp
        :param period_variable: when you go on the periodic
        :param threshold_variable: when you go on the threshold
        :param dp_variable:when you go on the dp
        :rtype: dict
        :return: the result of the strategy on the scenario
        """
        current_assets = self.optimal_weight  # initial weight
        row = {}
        sample_return = []
        tracking_error = 0.0
        transaction_cost = 0.0
        rebalanced_count = 0
        for ss in asset_returns_list:
            current_assets = [w*np.exp(r) for w, r in
                              zip(current_assets, ss)]  # port weights vary with market performance
            total = sum(current_assets)
            current_weight = [v / total for v in current_assets]  # varied portfolio weights
            weight_adjusted = self.optimal_weight  # target weight
            is_re_balanced = False
            if strategy == 'buy_and_hold':
                pass
            elif strategy == 'period':
                if (len(sample_return) + 1) % period_variable == 0:
                    is_re_balanced = True
            elif strategy == 'threshold':
                deviation = [abs(x - y) >= threshold_variable for x, y in zip(self.optimal_weight, current_weight)]
                is_re_balanced = any(deviation)
            elif strategy == 'dynamic_programming':
                current_equity_idx = round(round(current_weight[0], round(np.log10(1 / increment))) / increment)
                to_adjusted_equity = dp_variable[current_equity_idx]
                is_re_balanced = abs(to_adjusted_equity - current_equity_idx*increment) >= increment/10
                weight_adjusted = [to_adjusted_equity, round(1-to_adjusted_equity, round(np.log10(1/increment)))]
            else:
                raise Exception("strategies do not exist.")
            if is_re_balanced:  # time to re-balance
                rebalanced_count += 1
                diff = cost.cost_balancing(current_weight, weight_adjusted, one_transaction_cost)
                transaction_cost += sum([abs(x) * one_transaction_cost for x in diff])
                current_assets = [(w1 + w2) * total for w1, w2 in zip(current_weight, diff)]
                total = sum(current_assets)
                current_weight = weight_adjusted
            current_mean = sum(self.mean * current_weight)
            current_variance = float(np.dot(np.dot(current_weight, self.covariance_mat), current_weight))
            tracking_error += max(self.utility - utility(current_mean, current_variance, risk_aversion), 0.0)
            sample_return.append(total)
        row['# of Rebalanced'] = rebalanced_count
        row['Transaction Cost'] = transaction_cost
        row['Tracking Error Cost'] = tracking_error
        row['Total Cost'] = transaction_cost + tracking_error
        row['Return'] = (sample_return[-1]/sample_return[0] - 1)*100
        row['Sharpe Ratio'] = valuation.sharpe_ratio(sample_return, 10, risk_free=0.02)  # risk-free rate is 2%
        return row

    def buy_and_hold(self) -> pd.DataFrame:
        """
        strategy 1: buy-and-hold
        :return: result dataframe including # of rebalanced, transaction cost, sharpe ratio
        :rtype: pd.DataFrame
        """
        result_df = pd.DataFrame()
        row_idx = 0
        for s in self.sample:
            row_idx += 1
            row = self.a_scenario_run(s, strategy='buy_and_hold')
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[row_idx])])
        result_df.loc[-1] = result_df.mean(axis=0)  # dataframe column average
        print("###strategy: Buy and Hold")
        return result_df

    def periodic_rebalanced(self, frequency=12) -> pd.DataFrame:
        """
        strategy 2: rebalanced periodically
        :param frequency: re-balance frequency. default is 12 months
        :return: result dataframe including # of rebalanced, transaction cost, sharpe ratio
        :rtype: pd.DataFrame
        """
        result_df = pd.DataFrame()
        row_idx = 0
        for s in self.sample:
            row_idx += 1
            row = self.a_scenario_run(s, strategy='period', period_variable=frequency)
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[row_idx])])
        result_df.loc[-1] = result_df.mean(axis=0)
        print("###strategy: periodic_rebalanced, frequency %s months" % frequency)
        return result_df

    def threshold_rebalanced(self, tolerance=0.05) -> pd.DataFrame:
        """
        strategy 3: rebalanced with threshold
        :param tolerance: re-balance threshold. default is 5%
        :return: result dataframe including # of rebalanced, transaction cost, sharpe ratio
        :rtype: pd.DataFrame
        """
        result_df = pd.DataFrame()
        row_idx = 0
        for s in self.sample:
            row_idx += 1
            row = self.a_scenario_run(s, strategy='threshold', threshold_variable=tolerance)
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[row_idx])])
        result_df.loc[-1] = result_df.mean(axis=0)
        print("###strategy: threshold_rebalanced, threshold %s percents" % (tolerance*100))
        return result_df

    def dynamic_programming(self):
        """
        strategy 4: rebalanced with dynamic programming
        :return: result dataframe including # of rebalanced, transaction cost, sharpe ratio
        :rtype: pd.DataFrame
        """
        result_df = pd.DataFrame()
        row_idx = 0

        def modified_policy_iteration(repeat=20):
            """
            updating policy function and value(cost) function.

            reference:
            "Constructing Optimal Portfolio Rebalancing Strategies with a Two-Stage Multiresolution-Grid Model"
            :param repeat: the nmber of partial policy evaluation
            :return: policy function
            :rtype: list
            """
            length = (1+round(1/increment))
            value_func = [0.0]*length
            strategy = [self.optimal_weight[0]]*length  # policy function
            self.policy_basket = [strategy]
            sampling_length = 100_000
            sample = np.random.multivariate_normal(self.mean, self.covariance_mat, sampling_length)

            def transition_probability(equity_weight_before):  # O(n)
                """
                probability from 0weight1 to weight2 on asset price up_and_down.
                :param equity_weight_before: equity weight
                :return: 0~1 next weight probability
                :rtype: list
                """
                transition = [(np.exp(x[0])*equity_weight_before, np.exp(x[1])*(1-equity_weight_before)) for x in sample]
                transition_equity = [x[0]/sum(x) for x in transition]
                probabilities = [0]*(sampling_length+1)
                for p in transition_equity:
                    idx = round(p/increment)
                    probabilities[idx] += 1
                probabilities = [x/(sampling_length+1) for x in probabilities]
                return probabilities

            transition_probability_dict = {}  # Memorization to reduce calculation
            for i in range(length):
                tran_idx = round(i*increment, round(np.log10(1/increment)))
                transition_probability_dict[tran_idx] = transition_probability(tran_idx)

            tracking_err_dict = {}
            while True:
                # Partial Policy Evaluation
                value_func_tem = value_func
                for r in range(repeat):
                    for v in range(len(value_func_tem)):
                        equity_idx = round(v*increment, round(np.log10(1/increment)))
                        fixed_income_idx = round(1 - v*increment, round(np.log10(1/increment)))
                        w1 = [equity_idx, fixed_income_idx]
                        w2 = [strategy[v], round(1 - strategy[v], round(np.log10(1/increment)))]
                        if all([ww1 == ww2 for ww1, ww2 in zip(w1, w2)]):
                            transaction = 0.0
                        else:
                            balance = cost.cost_balancing(w1, w2, one_transaction_cost)
                            transaction = sum([abs(x)*one_transaction_cost for x in balance])
                        if strategy[v] not in tracking_err_dict.keys():
                            here_mean = sum(self.mean*w2)
                            here_variance = float(np.dot(np.dot(w2, self.covariance_mat), w2))
                            tracking_err_dict[strategy[v]] = max(self.utility - utility(here_mean, here_variance, risk_aversion), 0.0)
                        prob = transition_probability_dict[strategy[v]]
                        value_func_tem[v] = transaction + tracking_err_dict[strategy[v]] + dc_factor*sum([x*y for x, y in zip(prob, value_func_tem)])
                # Policy Improvement
                strategy_updated = []
                value_func_updated = []
                for v1 in range(len(value_func_tem)):
                    equity_idx = round(v1*increment, round(np.log10(1/increment)))
                    fixed_income_idx = round(1 - v1*increment, round(np.log10(1/increment)))
                    one = []
                    w1 = [equity_idx, fixed_income_idx]
                    for v2 in range(len(value_func_tem)):
                        equity_idx2 = round(v2*increment, round(np.log10(1/increment)))
                        fixed_income_idx2 = round(1 - v2*increment, round(np.log10(1/increment)))
                        w2 = [equity_idx2, fixed_income_idx2]
                        if all([ww1 == ww2 for ww1, ww2 in zip(w1, w2)]):
                            transaction = 0.0
                        else:
                            balance = cost.cost_balancing(w1, w2, one_transaction_cost)
                            transaction = sum([abs(x)*one_transaction_cost for x in balance])
                        if equity_idx2 not in tracking_err_dict.keys():
                            here_mean = sum(self.mean * w2)
                            here_variance = float(np.dot(np.dot(w2, self.covariance_mat), w2))
                            tracking_err_dict[equity_idx2] = max(self.utility - utility(here_mean, here_variance, risk_aversion), 0.0)
                        prob = transition_probability_dict[equity_idx2]
                        one.append(transaction + tracking_err_dict[equity_idx2] + dc_factor*sum([x*y for x, y in zip(prob, value_func_tem)]))
                    strategy_updated.append(round(np.argmin(one)*increment, round(np.log10(1/increment))))
                    value_func_updated.append(min(one))
                comparison1 = max([abs(x2 - x1) for x1, x2 in zip(value_func, value_func_updated)])
                comparison2 = [x2 == x1 for x1, x2 in zip(strategy, strategy_updated)]
                value_func = value_func_updated
                strategy = strategy_updated
                self.policy_basket.append(strategy)
                condition1 = comparison1 <= dp_tol*(1-dc_factor)/dc_factor
                condition2 = all(comparison2)
                if condition1 or condition2:
                    if condition1:
                        print("the difference between cost function and the updated has become small enough.")
                    if condition2:
                        print("the policy function has converged.")
                    break
            return strategy

        dp_strategy = modified_policy_iteration()
        for s in self.sample:
            row_idx += 1
            row = self.a_scenario_run(s, strategy='dynamic_programming', dp_variable=dp_strategy)
            result_df = pd.concat([result_df, pd.DataFrame(row, index=[row_idx])])
        result_df.loc[-1] = result_df.mean(axis=0)
        return result_df


# Results
port = Portfolio(dt.datetime.strptime('2011-02-01', '%Y-%m-%d'), dt.datetime.strptime('2024-04-09', '%Y-%m-%d'), 1000, 10)
print("Optimal Portfolio: ", port.optimal_weight)
print(port.buy_and_hold().loc[-1])
print(port.dynamic_programming().loc[-1])
pd.DataFrame(port.policy_basket).to_csv('policy.csv')
print(port.periodic_rebalanced(frequency=1).loc[-1])
print(port.periodic_rebalanced(frequency=3).loc[-1])
print(port.periodic_rebalanced(frequency=12).loc[-1])
print(port.periodic_rebalanced(frequency=24).loc[-1])
print(port.threshold_rebalanced(tolerance=0.01).loc[-1])
print(port.threshold_rebalanced(tolerance=0.03).loc[-1])
print(port.threshold_rebalanced(tolerance=0.05).loc[-1])
print(port.threshold_rebalanced(tolerance=0.10).loc[-1])
print("############################")

periodic_dict = {}
tolerance_dict = {}
basic_total_cost = float(port.buy_and_hold().iloc[-1]['Total Cost'])*100
basic_tc = float(port.buy_and_hold().iloc[-1]['Transaction Cost'])*100
basic_trc = float(port.buy_and_hold().iloc[-1]['Tracking Error Cost'])*100
basic_sharpe = port.buy_and_hold().iloc[-1]['Sharpe Ratio']

for t in range(12*3):  # one to 36 months and 36 percents
    imported_periodic = port.periodic_rebalanced(frequency=t+1)
    imported_tolerance = port.threshold_rebalanced(tolerance=(t+1)/100)
    periodic_dict[t+1] = [float(imported_periodic.iloc[-1]['Total Cost'])*100, float(imported_periodic.iloc[-1]['Transaction Cost'])*100, float(imported_periodic.iloc[-1]['Tracking Error Cost'])*100, imported_periodic.iloc[-1]['Sharpe Ratio']]
    tolerance_dict[t+1] = [float(imported_tolerance.iloc[-1]['Total Cost'])*100, float(imported_tolerance.iloc[-1]['Transaction Cost'])*100, float(imported_tolerance.iloc[-1]['Tracking Error Cost'])*100, imported_tolerance.iloc[-1]['Sharpe Ratio']]
periodic_dict['BH'] = [basic_total_cost, basic_tc, basic_trc, basic_sharpe]
tolerance_dict['BH'] = [basic_total_cost, basic_tc, basic_trc, basic_sharpe]
pd.DataFrame.from_dict(data=periodic_dict, orient='index', columns=['Total Cost', 'Transaction Cost', 'Tracking Error Cost', 'Sharpe Ratio']).to_csv('periodic.csv')
pd.DataFrame.from_dict(data=tolerance_dict, orient='index', columns=['Total Cost', 'Transaction Cost', 'Tracking Error Cost', 'Sharpe Ratio']).to_csv('tolerance.csv')


