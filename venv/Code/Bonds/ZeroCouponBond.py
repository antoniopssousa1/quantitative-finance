class ZeroCouponBond:
    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate / 100  # Convert percentage to decimal

    def present_value(self,x,n):
        return x / (1 + self.interest_rate) ** n

    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)

    def yield_to_maturity(self):
        return (self.principal / self.calculate_price()) ** (1 / self.maturity) - 1


if __name__ == "__main__":

    bond = ZeroCouponBond(principal=1000, maturity=2, interest_rate=4)
    price = bond.calculate_price()
    ytm = bond.yield_to_maturity()

    print(f"Price of the zero-coupon bond: ${price:.2f}")
    print(f"Yield to Maturity: {ytm:.2%}")