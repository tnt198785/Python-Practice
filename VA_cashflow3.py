# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:11:15 2024

@author: Zhenxian Gong
"""
import numpy as np
import pandas as pd
from functools import wraps
from datetime import date
from dateutil.relativedelta import relativedelta




class VariableAnnuity:
    def __init__(self, initial_investment, 
                 annuitant_age,
                 issue_date,
                 first_withdraw_age, 
                 commencement_age,
                 death_age,
                 mortality_expense,
                 fund_fees,
                 rider_charge,
                 withdraw_rate,
                 set_up,
                 set_up_period,
                 fixed_funds_auto_re_target,
                 max_annual_wd_table):
        self.initial_investment = initial_investment
        self.annuitant_age = annuitant_age
        self.issue_date = issue_date
        self.first_withdraw_age = first_withdraw_age
        self.commencement_age = commencement_age
        self.death_age = death_age
        self.mortality_expense = mortality_expense
        self.fund_fees = fund_fees
        self.rider_charge = rider_charge
        self.withdraw_rate = withdraw_rate
        self.set_up = set_up
        self.set_up_period = set_up_period
        self.target = fixed_funds_auto_re_target
        self.max_annual_wd_table = max_annual_wd_table

        self.contribution = []
        self.mortality_table = {}
        self.__cache = {}

  
    def cache_result(func):
        @wraps(func)
        def wrapper_cache(*args, **kwargs):
            self = args[0]
            if func.__name__ not in self.__cache.keys():
                self.__cache[func.__name__] = dict()
            cache_key = (args[1:] + tuple(kwargs.items()))
            if cache_key in self.__cache[func.__name__].keys():
                result = self.__cache[func.__name__][cache_key]
            else:
                result = func(*args, **kwargs)
                self.__cache[func.__name__][cache_key] = result
            return result

        return wrapper_cache

    def mortality_rate(age, mortality_table):
        row_index = mortality_table.index[mortality_table['Age'] == age]
        mortality = mortality_table.iloc[row_index]['Mortality_Rate'].values[0]
        return mortality
    # @cache_result
    def maximum_annual_withdraw_rate(self, year, current_age):
        ages = self.max_annual_wd_table['Ages']
        rates = self.max_annual_wd_table['Rates']
        if self._growth(year):
            wd_rate = 0
        else:
            for i in range(len(ages)-1):
                if current_age >= ages[i] and current_age < ages[i+1]:
                    wd_rate = rates[i]
                elif current_age >= ages[-1]:
                    wd_rate = rates[-1]
        return wd_rate
    # @cache_result
    def fund_return(self, fund_return_scen):
        self.fund_1_return = fund_return_scen['Fund_1_Return']
        self.fund_2_return = fund_return_scen['Fund_2_Return']
        return self.fund_1_return, self.fund_2_return
    # @cache_result
    def schedule_proj(self):
        num_step = self.death_age - self.annuitant_age
        self.age_proj = self.annuitant_age + np.array([ i for i in range(num_step + 1)])
        self.year_proj = np.arange(num_step + 1)
        self.anniversary_proj = np.array([self.issue_date + relativedelta(years = i) for i in range(num_step + 1)])
        return self.age_proj, self.anniversary_proj, self.year_proj
    
    @cache_result
    def _fund_1_pre_fee(self, year):
        if year == 0:
            return self.initial_investment * 0.16
        else:
            return self._fund_1_post_re(year - 1) * (1 + self.fund_1_return[year])
    @cache_result
    def _fund_2_pre_fee(self, year):
        if year == 0:
            return self.initial_investment * 0.64
        else:
            return self._fund_2_post_re(year - 1) * (1 + self.fund_2_return[year])    
    @cache_result
    def _av_pre_fee(self, year):
        if year == 0:
            return self._fund_1_pre_fee(0) + self._fund_2_pre_fee(0)
        else:
            return self._fund_1_pre_fee(year) + self._fund_2_pre_fee(year)
    @cache_result
    def _fund_1_post_re(self, year):
        if year == 0:
            return self._fund_1_post_death_claims(0)
        else:
            if self._rebalance(year):
                return self._av_post_death(year) * self.target
            else:
                return self._fund_1_post_death_claims(year)
    @cache_result  
    def _fund_2_post_re(self, year):
       if year == 0:
            return self._av_post_charges(0) - self._fund_1_post_re(0)
       else:
            return self._av_post_charges(year) - self._fund_1_post_re(year)
    @cache_result
    def _fund_1_post_death_claims(self, year):
        if year == 0:
            if self._av_post_death(0) == 0:
                return 0
            else:
                return self._fund_1_post_charges(0)*(self._av_post_death(0) / self._av_post_charges(0))
        else:
            if self._av_post_death(year) == 0:
                return 0
            else:
                return self._fund_1_post_charges(year)*(self._av_post_death(year) / self._av_post_charges(year))
    @cache_result
    def _fund_2_post_death_claims(self, year):
        if year == 0:
            if self._av_post_death(0) == 0:
                return 0
            else:
                return self._fund_2_post_charges(0)*(self._av_post_death(0) / self._av_post_charges(0))  
        else:      
            if self._av_post_death(year) == 0:
                return 0
            else:
                return self._fund_2_post_charges(year)*(self._av_post_death(year) / self._av_post_charges(year))        
            
    @cache_result
    def _fund_1_post_charges(self, year):
        if year == 0:
            if self._av_post_charges(0) == 0:
                return 0
            else:
                return self._fund_1_post_withdraw(0) * (self._av_post_charges(0)/self._av_post_withdraw(0))
        else:
            if self._av_post_charges(year) == 0:
                return 0
            else:
                return self._fund_1_post_withdraw(year) * (self._av_post_charges(year)/self._av_post_withdraw(year))
    @cache_result
    def _fund_2_post_charges(self, year):
        if year == 0:
            if self._av_post_charges(0) == 0:
                return 0
            else:
            
                return self._fund_2_post_withdraw(0) * (self._av_post_charges(0)/self._av_post_withdraw(0))
        else:
            if self._av_post_charges(year) == 0:
                return 0
            else:
                
                return self._fund_2_post_withdraw(year) * (self._av_post_charges(year)/self._av_post_withdraw(year))
    @cache_result
    def _fund_1_post_withdraw(self, year):
        if year == 0:
            if self._av_post_withdraw(0) == 0:
                return 0
            else:
                return self._fund_1_pre_withdraw(0) * (self._av_post_withdraw(0)/self._av_pre_withdraw(0))
        else:
            if self._av_post_withdraw(year) == 0:
                return 0
            else:
                return self._fund_1_pre_withdraw(year) * (self._av_post_withdraw(year)/self._av_pre_withdraw(year))
    @cache_result
    def _fund_2_post_withdraw(self, year):
        if year == 0:
            if self._av_post_withdraw(0) == 0:
                return 0
            else:
                return self._fund_2_pre_withdraw(0) * (self._av_post_withdraw(0)/self._av_pre_withdraw(0))
        else:
            if self._av_post_withdraw(year) == 0:
                return 0
            else:
                return self._fund_2_pre_withdraw(year) * (self._av_post_withdraw(year)/self._av_pre_withdraw(year))
            
    @cache_result
    def _fund_1_pre_withdraw(self, year):
        if year == 0:
            if self._av_pre_withdraw(0) == 0:
                return 0
            else:
                return self._fund_1_pre_fee(0) * (self._av_pre_withdraw(0)/self._av_pre_fee(0))
        else:
            if self._av_pre_withdraw(year) == 0:
                return 0
            else:
                return self._fund_1_pre_fee(year) * (self._av_pre_withdraw(year)/self._av_pre_fee(year))
    @cache_result
    def _fund_2_pre_withdraw(self, year):
        if year == 0:
            if self._av_pre_withdraw(0) == 0:
                return 0
            else:
                return self._fund_2_pre_fee(0) * (self._av_pre_withdraw(0)/self._av_pre_fee(0))  
        else:     
            if self._av_pre_withdraw(year) == 0:
                return 0
            else:
                return self._fund_2_pre_fee(year) * (self._av_pre_withdraw(year)/self._av_pre_fee(year))        
    @cache_result
    def _rider_charges(self, year):
        if year == 0:
            return 0
        else:
            return self.rider_charge * self._av_post_withdraw(year)
    @cache_result
    def _fees(self, year):
        if year == 0:
            return 0
        else:
            return (self.fund_fees + self.mortality_expense) * self._av_post_death(year - 1)
        
    @cache_result
    def _av_post_charges(self, year):
        if year == 0:
            return self._av_pre_withdraw(0)
        else:
            return self._av_post_withdraw(year) -self._rider_charges(year)
       
    @cache_result
    def _av_pre_withdraw(self, year):
        contribution = self.contribution[year]
        if year == 0:
            return self._av_pre_fee(year) + contribution - self._fees(year)
        else:    
            return max(0, self._av_pre_fee(year) + contribution - self._fees(year))
        
    @cache_result
    def _av_post_withdraw(self, year):
        if year == 0:
            return self._av_pre_withdraw(year)
        else:
            return max(0, self._av_pre_withdraw(year) - self._withdraw(year))
    @cache_result
    def _withdraw(self, year):
        if self._withdraw_period(year):
            return self.withdraw_rate * self._withdraw_base(year)
        else:
            if self._automatic_periodic_benefit_status(year):
                return self._max_annual_withdraw(year)
            else:
                return 0
    @cache_result   
    def _av_post_death(self, year):
        if year == 0:
            return self._av_pre_withdraw(0)
        else:
            return max(self._av_post_charges(year) - self._death_pay(year), 0)
        
    @cache_result
    def _death_pay(self, year):
        mortality_table = self.mortality_table
        if year == 0:
            return 0
        else:
            if not (self._growth(year) or self._withdraw_period(year) or self._automatic_periodic_benefit_status(year) or self._last_death(year)):
                return 0
            else:
                mortality_rate = VariableAnnuity.mortality_rate(self.age_proj[year], mortality_table)
                return max(self._death_benefit(year-1), self._rop_death_base(year-1)) * mortality_rate

    @cache_result
    def _rop_death_base(self, year):
        mortality_table = self.mortality_table
        if year == 0:
            return self.initial_investment
            
        else:
            mortality_rate = VariableAnnuity.mortality_rate(self.age_proj[year], mortality_table)
            return self._rop_death_base(year - 1) * (1 - mortality_rate)
            
    @cache_result
    def _nar_death_claims(self, year):
        mortality_table = self.mortality_table
        if year == 0:
            return 0
        else:
            return max(0, self._death_pay(year) - self._av_post_charges(year))
        
    @cache_result
    def _death_benefit(self, year):
        mortality_table = self.mortality_table
        contribution = self.contribution[year]
        if year == 0:
           return self.initial_investment
        else:
            mortality_rate = VariableAnnuity.mortality_rate(self.age_proj[year], mortality_table)
            return max(0, self._death_benefit(year - 1) * (1 - mortality_rate) + contribution - self._fees(year) - self._withdraw(year - 1) - self._rider_charges(year))
    @cache_result
    def _withdraw_base(self, year):
        mortality_table = self.mortality_table
        contribution = self.contribution[year]
        if year == 0:
            return self.initial_investment
            
        else:
            
            mortality_rate = VariableAnnuity.mortality_rate(self.age_proj[year], mortality_table)
            if self._growth(year):
                Growth_year = self._av_post_death(year)
            else:
                Growth_year = 0
            Withdraw_surv_year = self._withdraw_base(year - 1) * (1 - mortality_rate) + contribution
            if self._eligible_step_up(year):
                Withdraw_set_up_year = self._withdraw_base(year - 1) * (1 - mortality_rate) * (1 + self.set_up) + contribution - self._fees(year) - self._rider_charges(year)
            else:
                Withdraw_set_up_year = 0
        return max(Growth_year, Withdraw_surv_year, Withdraw_set_up_year)
           
    @cache_result    
    def _max_annual_withdraw_rate(self, year):
        if year == 0:
            return 0
        else:
            age = self.age_proj[year]
            return self.maximum_annual_withdraw_rate(year, age)

    @cache_result
    def _max_annual_withdraw(self, year):
        return self._withdraw_base(year) * self._max_annual_withdraw_rate(year)
        
    @cache_result
    def _eligible_step_up(self, year):
        
        if year == 0:
            return False
        else:
            if self._growth(year) and year <= self.set_up_period:
                return True
            else:
                return False
    @cache_result
    def _growth(self, year):
        age = self.age_proj[year]
        if age == self.age_proj[0]:
            return False
        else:
            if age <= self.first_withdraw_age and age <= self.commencement_age and age < self.death_age:
                return True
            else:
                return False
    @cache_result            
    def _withdraw_period(self, year):
        if year == 0:
            return False
        else:
            age = self.age_proj[year]
            if age > self.first_withdraw_age and self._av_post_death(year - 1) > 0 and age < self.death_age:
                return True
            else:
                return False
            
    @cache_result
    def _automatic_periodic_benefit_status(self, year):
        age = self.age_proj[year]
        if year == 0:
            return False
        elif year == 1:
            return False
        else:
            if age >=self.death_age:
                return False
            else:
                if self._withdraw_period(year - 1) and self._av_post_death(year - 1) == 0:
                    return True
                else:
                    return self._automatic_periodic_benefit_status(year - 1)
    @cache_result             
    def _last_death(self, year):
        age = self.age_proj[year]
        if age == self.death_age:
            return True
        else:
            return False
    @cache_result
    def _rebalance(self, year):
        if year == 0:
            return False
        else:
            return bool(self._withdraw_period(year) or self._automatic_periodic_benefit_status(year))
        
    
    def va_cashflow(self, fund_return_scen):
        
        self.fund_return(fund_return_scen)

        self.schedule_proj()
        years = self.year_proj
        anniversary = self.anniversary_proj
        age = self.age_proj
        fund_1_pre_fee = []
        fund_2_pre_fee = []
        av_pre_fee = []
        fees = []
        av_pre_withdraw = []
        fund_1_pre_winthdraw = []
        fund_2_pre_winthdraw = []
        withdraw_amount = []
        av_post_withdraw = []
        fund_1_post_withdraw = []
        fund_2_post_withdraw = []
        rider_charges = []
        av_post_charges = []
        fund_1_post_charges = []
        fund_2_post_charges = []
        death_pay = []
        av_post_death = []
        fund_1_post_death = []
        fund_2_post_death = []
        fund_1_post_rebalance = []
        fund_2_post_rebalance = []
        rop_death_base = []
        nar_death_claims = []
        death_benefit = []
        withdraw_base = []
        max_annual_withdraw = []
        max_annual_withdraw_rate = []
        eligible_step_up = []
        growth_phase = []
        withdraw_phase = []
        automatic_periodic_benefit_status = []
        last_death = []
        rebalance_indicator = []
        for year in self.year_proj:

            fund_1_pre_fee.append(self._fund_1_pre_fee(year))
            fund_2_pre_fee.append(self._fund_2_pre_fee(year))
            av_pre_fee.append(self._av_pre_fee(year))
            fees.append(self._fees(year))
            fund_1_pre_winthdraw.append(self._fund_1_pre_withdraw(year))
            fund_2_pre_winthdraw.append(self._fund_2_pre_withdraw(year))
            av_post_withdraw.append(self._av_post_withdraw(year))
            fund_1_post_withdraw.append(self._fund_1_post_withdraw(year))
            fund_2_post_withdraw.append(self._fund_2_post_withdraw(year))
            rider_charges.append(self._rider_charges(year))
            fund_1_post_charges.append(self._fund_1_post_charges(year))
            fund_2_post_charges.append(self._fund_2_post_charges(year))
            fund_1_post_death.append(self._fund_1_post_death_claims(year))
            fund_2_post_death.append(self._fund_2_post_death_claims(year))
            fund_1_post_rebalance.append(self._fund_1_post_re(year))
            fund_2_post_rebalance.append(self._fund_2_post_re(year))
            rop_death_base.append(self._rop_death_base(year))
            nar_death_claims.append(self._nar_death_claims(year))
            death_benefit.append(self._death_benefit(year))
            withdraw_base.append(self._withdraw_base(year))
             
            max_annual_withdraw_rate.append(self._max_annual_withdraw_rate(year))
            last_death.append(self._last_death(year))
            withdraw_amount.append(self._withdraw(year))
            max_annual_withdraw.append(self._max_annual_withdraw(year))
            eligible_step_up.append(self._eligible_step_up(year))
            growth_phase.append(self._growth(year))
            
            
            av_pre_withdraw.append(self._av_pre_withdraw(year)) 
            av_post_charges.append(self._av_post_charges(year))
            death_pay.append(self._death_pay(year))
            av_post_death.append(self._av_post_death(year))

            withdraw_phase.append(self._withdraw_period(year))
            rebalance_indicator.append(self._rebalance(year))
            automatic_periodic_benefit_status.append(self._automatic_periodic_benefit_status(year))
            
        cumulative_withdraw = np.cumsum(withdraw_amount)
        df = (1 + self.fund_1_return) ** (-years)
        pv_db_claims = sum(nar_death_claims * df)
        withdraw_claims = np.maximum(0, np.array(withdraw_amount)[1:] - np.array(av_post_death)[:-1])
        withdraw_claims = np.insert(withdraw_claims, 0, 0)
        pv_wb_claim = sum(withdraw_claims * df)
        pv_rc = sum(rider_charges * df)
        output1 = pd.DataFrame({'Year': years,
                               'Anniversary': anniversary,
                               'Age': age,
                               'Contribution': self.contribution,
                               'AV Pre-Fee': av_pre_fee,
                               'Fund 1 Pre-Fee': fund_1_pre_fee,
                               'Fund 2 Pre-Fee': fund_2_pre_fee,
                               'M&E/Fund Fees': fees,
                               'AV Pre-Withdraw': av_pre_withdraw,
                               'Fund 1 Pre-Withdraw': fund_1_pre_winthdraw,
                               'Fund 2 Pre_withdraw': fund_2_pre_winthdraw,
                               'Withdraw Amount': withdraw_amount,
                               'AV Post-Withdraw': av_post_withdraw,
                               'Fund 1 Post-Withdraw': fund_1_post_withdraw,
                               'FUnd 2 Post-Withdraw': fund_2_post_withdraw,
                               'Rider Charge': rider_charges,
                               'AV Post-Charges': av_post_charges,
                               'Fund 1 Post-Charges': fund_1_post_charges,
                               'Fund 2 Post-Charges': fund_2_post_charges,
                               'Death Payments': death_pay,
                               'AV Post-Death Claims': av_post_death,
                               'Fund 1 Post-Death Claims': fund_1_post_death,
                               'Fund 2 Post-Death Claims': fund_2_post_death,
                               'Fund 1 Post-Rebalance': fund_1_post_rebalance,
                               'Fund 2 Post-Rebalance': fund_2_post_rebalance,
                               'ROP Death Base': rop_death_base,
                               'NAR Death Claims': nar_death_claims,
                               'Death Benefit Base': death_benefit,
                               'Withdraw Base': withdraw_base,
                               'Cumulative Withdraw': cumulative_withdraw,
                               'Maximum Annual Withdraw': max_annual_withdraw,
                               'Maximum Annuam Withdraw Rate': max_annual_withdraw_rate,
                               'Eligible Step Up': eligible_step_up,
                               'Growth Phase': growth_phase,
                               'Withdraw Phase': withdraw_phase,
                               'Automatic Periodic Benefit Status': automatic_periodic_benefit_status,
                               'Last Death': last_death,
                               'Fund 1 Return': self.fund_1_return,
                               'Fund 2 Return': self.fund_2_return,
                               'Rebalance Indicator': rebalance_indicator,
                               'DF': df,
                               })
        output2 = pd.DataFrame({'PV DB Claim': pv_db_claims,
                               'PV WB Claim': pv_wb_claim,
                               'PV RC': pv_rc 
                               }, index = [0])
        
        with pd.ExcelWriter("VA CashFlow Python Replica.xlsx") as writer:
            output1.to_excel(writer, sheet_name="VA Cashflow", index=False)
            output2.to_excel(writer, sheet_name="Present Values")
           
       
    
    


        
                

        