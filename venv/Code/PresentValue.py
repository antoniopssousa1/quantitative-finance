import math

def future_value_discrete(pv, r, n):
    return pv * (1 + r) ** n

def present_value_discrete(fv, r, n):
    return fv / (1 + r) ** n 

def present_value_continuous(fv, r, n):
    return fv * math.exp(-r * n)

def future_value_continuous(pv, r, n):
    return pv * math.exp(r * n)

 