import numpy as np
import pandas as pd
from datetime import date
from VA_cashflow3 import VariableAnnuity
import sys


np.random.seed(10)

initial_payment = 100000 
annuitant_age = 60
issue_date = date(2016, 8, 1)
first_withdraw = 70
commencement_age = 80
death_age = 100
mortality_expense = 0.014
fund_fees = 0.0015
rider_charge = 0.0085
withdraw_rate = 0.03
set_up = 0.06
set_up_period = 10
fixed_funds_rebalancing = 0.2
max_annual_wd_table = {'Ages': [59.5, 65, 76, 80],
                        'Rates': [0.04, 0.05, 0.06, 0.07]}
contributions = np.zeros(death_age - annuitant_age + 1)
risk_free_rates = 0.03 * np.ones(death_age - annuitant_age)
stochastic_rate = np.log(1 + risk_free_rates) - 0.5 * 0.16 ** 2 + 0.16 * np.random.standard_normal(len(risk_free_rates))
stochastic_return = np.exp(stochastic_rate) - 1
risk_free_rates = np.insert(risk_free_rates, 0 , 0)
stochastic_return = np.insert(stochastic_return, 0, 0)
fund_returns = {'Fund_1_Return': risk_free_rates,
               'Fund_2_Return': stochastic_return}
mortality_age = np.linspace(0, 115, 116)
mortality_rates = 0.005 * np.ones(len(mortality_age))
mortality_table = pd.DataFrame({'Age': mortality_age,
                   'Mortality_Rate': mortality_rates})

va = VariableAnnuity(initial_payment, 
                 annuitant_age,
                 issue_date,
                 first_withdraw, 
                 commencement_age,
                 death_age,
                 mortality_expense,
                 fund_fees,
                 rider_charge,
                 withdraw_rate,
                 set_up,
                 set_up_period,
                 fixed_funds_rebalancing,
                 max_annual_wd_table)
va.contribution = contributions
va.mortality_table = mortality_table

print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)
va.va_cashflow(fund_returns)




       
