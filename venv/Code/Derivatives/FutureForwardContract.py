import numpy as np


class ForwardFutureContract:
    def __init__(self, spot_price, risk_free_rate, maturity,
                 position="long", delivery_price=None):
        # spot_price S(t)
        self.S = spot_price
        # risk_free_rate: r (continuous compounding)
        self.r = risk_free_rate
        # maturity: time to maturity (T - t)
        self.T = maturity
        # long or short
        self.position = position.lower()
        # delivery_price (futures price at t=0)
        self.K = delivery_price

    # forward price = futures price (with no dividends and no storage)
    def forward_price(self):
        return self.S * np.exp(self.r * self.T)

    def contract_value(self):
        if self.K is None:
            raise ValueError("Delivery price K must be specified.")

        # V(t) = S(t) - K exp ^ {-r(T - t)}
        V = self.S - self.K * np.exp(-self.r * self.T)

        # futures and forwards are linear instruments: the short is -long
        if self.position == "short":
            V = -V

        return V


contract = ForwardFutureContract(
    spot_price=100,
    risk_free_rate=0.05,
    maturity=1,
    position="long",
    delivery_price=105
)

contract.K = contract.forward_price()

print(f"Forward price: ${contract.forward_price():.2f}")
print(f"Contract value: ${contract.contract_value():.2f}")