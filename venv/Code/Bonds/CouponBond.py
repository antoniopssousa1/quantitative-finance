class CouponBond:


    def __init__(self, principal, coupon_rate, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.coupon_rate = coupon_rate / 100  # Convert percentage to decimal
        self.interest_rate = interest_rate / 100  # Convert percentage to decimal

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n

    def calculate_price(self):
        price = 0
        for t in range(1, self.maturity + 1):
            price += self.present_value(self.principal * self.coupon_rate, t)
        price += self.present_value(self.principal, self.maturity)
        return price
    
# Example usage:

if __name__ == "__main__":
    bond = CouponBond(principal=1000, coupon_rate=10, maturity=3, interest_rate=4)
    price = bond.calculate_price()
    print(f"The price of the coupon bond is: ${price:.2f}")
