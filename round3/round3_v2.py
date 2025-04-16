from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from collections import deque
from math import log, sqrt, exp, isclose
from statistics import NormalDist

TIME_TO_EXPIRY = 248/252 ##CHANGE BEFORE SUBMISSION

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 45,
    },
    Product.KELP: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.4776,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.15,
        "disregard_edge": 1,
        "join_edge": 3,
        "default_edge": 2,
        "soft_position_limit": 30,
    },
    Product.SPREAD1: {
        "default_spread_mean": 10,
        "default_spread_std": 85,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 10,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 75,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "ivolatility": 0.16,
        "delta": 0.5,
        "gamma": 0.1,
        "target_position": 0,
        "join_edge": 5,
        "disregard_edge": 0,
        "default_edge": 0,
        "take_width": 1,
        "strike": 9500
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "ivolatility": 0.16,
        "delta": 0.5,
        "gamma": 0.1,
        "target_position": 0,
        "join_edge": 3,
        "disregard_edge": 0,
        "default_edge": 0,
        "take_width": 1,
        "strike": 9750
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "ivolatility": 0.16,
        "delta": 0.5,
        "gamma": 0.1,
        "target_position": 0,
        "join_edge": 3,
        "disregard_edge": 0.5,
        "default_edge": 4,
        "take_width": 0.5,
        "strike": 10000
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "ivolatility": 0.16,
        "delta": 0.5,
        "gamma": 0.1,
        "target_position": 0,
        "join_edge": 1.5,
        "disregard_edge": 1,
        "default_edge": 4,
        "take_width": 0.5,
        "strike": 10250
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "ivolatility": 0.2,
        "delta": 0.5,
        "gamma": 0.1,
        "target_position": 0,
        "join_edge": 1.5,
        "disregard_edge": 1,
        "default_edge": 4,
        "take_width": 1,
        "strike": 10500
    },
}
    

BASKET_WEIGHTS_1 = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}
BASKET_WEIGHTS_2 = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}
VOLCANIC_VOUCHERS = [
    Product.VOLCANIC_ROCK_VOUCHER_9500,
    Product.VOLCANIC_ROCK_VOUCHER_9750,
    Product.VOLCANIC_ROCK_VOUCHER_10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250,
    Product.VOLCANIC_ROCK_VOUCHER_10500
]

class BlackScholesGreeks:
    @staticmethod
    def _compute_shared(spot, strike, T, vol, r=0.0, q=0.0):
        """Precompute shared values for Greeks and price"""
        sqrt_T = sqrt(T)
        log_SK = log(spot / strike)
        drift_adj = (r - q + 0.5 * vol**2) * T
            
        
        d1 = (log_SK + drift_adj) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T
        
        pdf_d1 = NormalDist().pdf(d1)
        cdf_d1 = NormalDist().cdf(d1)
        cdf_d2 = NormalDist().cdf(d2)
        
        return {
            'd1': d1,
            'd2': d2,
            'pdf_d1': pdf_d1,
            'cdf_d1': cdf_d1,
            'cdf_d2': cdf_d2,
            'sqrt_T': sqrt_T,
            'discount': exp(-r * T),
            'div_discount': exp(-q * T)
        }

    @staticmethod
    def call_price(spot, strike, T, vol, r=0.0, q=0.0):
        shared = BlackScholesGreeks._compute_shared(spot, strike, T, vol, r, q)
        return (spot * shared['div_discount'] * shared['cdf_d1'] 
                - strike * shared['discount'] * shared['cdf_d2'])

    @staticmethod
    def delta(spot, strike, T, vol, r=0.0, q=0.0):
        shared = BlackScholesGreeks._compute_shared(spot, strike, T, vol, r, q)
        return shared['div_discount'] * shared['cdf_d1']

    @staticmethod
    def gamma(spot, strike, T, vol, r=0.0, q=0.0):
        shared = BlackScholesGreeks._compute_shared(spot, strike, T, vol, r, q)
        return (shared['div_discount'] * shared['pdf_d1'] 
                / (spot * vol * shared['sqrt_T']))

    @staticmethod
    def vega(spot, strike, T, vol, r=0.0, q=0.0):
        shared = BlackScholesGreeks._compute_shared(spot, strike, T, vol, r, q)
        return spot * shared['div_discount'] * sqrt(T) * shared['pdf_d1']

    @staticmethod
    def implied_volatility_newton(market_price, spot, strike, T, r=0.0, q=0.0):
        """Newton-Raphson with analytical vega and value reuse"""
        def f(sigma):
            # Reuse all precomputed values for Greeks
            
            shared = BlackScholesGreeks._compute_shared(spot, strike, T, sigma, r, q)
            price = (spot * shared['div_discount'] * shared['cdf_d1'] 
                     - strike * shared['discount'] * shared['cdf_d2'])
            vega = spot * shared['div_discount'] * shared['sqrt_T'] * shared['pdf_d1']
            return price - market_price, vega

        try:
            # Use Newton with analytical derivative
            return newton(
                func=lambda x: f(x)[0],
                x0=0.20,
                fprime=lambda x: f(x)[1],
                tol=1e-8,
                maxiter=50
            )
        except RuntimeError:
            return float('nan')

    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholesGreeks.call_price(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.SYNTHETIC1: 60,
            Product.SYNTHETIC2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            }
        self.window_size = 10
        self.ink_history  = deque(maxlen=self.window_size)
        self.spike_duration = deque(maxlen=3)

    def detect_spike(self, current_price):
        if len(self.ink_history) < 5: 
            return False
        mean = np.mean(list(self.ink_history)[-5:])
        flag =  abs(current_price - mean) > 3 * np.std(list(self.ink_history))
        if flag:
            print("Spike detected!", self.spike_duration)
        return flag
            

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None
      
    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                if len(self.ink_history) >= self.window_size:
                    returns = np.diff(list(self.ink_history))/ list(self.ink_history)[:-1]
                    volatility = np.std(returns)
                    #print("Volatility: ", volatility)
                    dynamic_beta = self.params[Product.SQUID_INK]["reversion_beta"] *(1+volatility* 1e2)
                    pred_returns = last_returns * dynamic_beta
                else:
                    pred_returns = (
                        last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                    )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None
    
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
        spike = False,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        if spike and product == Product.SQUID_INK:
            take_width *= 3

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, OrderDepth]:
        
        synthetic_order_price = {
            Product.SYNTHETIC1: OrderDepth(),
            Product.SYNTHETIC2: OrderDepth(),
        }
        
        for product,basket in [( Product.SYNTHETIC1,BASKET_WEIGHTS_1), (Product.SYNTHETIC2, BASKET_WEIGHTS_2)]:
            CROISSANTS_PER_BASKET = basket[Product.CROISSANTS]
            JAMS_PER_BASKET = basket[Product.JAMS]
            DJEMBES_PER_BASKET = basket.get(Product.DJEMBES, 0)

            # Calculate the best bid and ask for each component
            CROISSANTS_best_bid = (
                max(order_depths[Product.CROISSANTS].buy_orders.keys())
                if order_depths[Product.CROISSANTS].buy_orders
                else 0
            )
            CROISSANTS_best_ask = (
                min(order_depths[Product.CROISSANTS].sell_orders.keys())
                if order_depths[Product.CROISSANTS].sell_orders
                else float("inf")
            )
            JAMS_best_bid = (
                max(order_depths[Product.JAMS].buy_orders.keys())
                if order_depths[Product.JAMS].buy_orders
                else 0
            )
            JAMS_best_ask = (
                min(order_depths[Product.JAMS].sell_orders.keys())
                if order_depths[Product.JAMS].sell_orders
                else float("inf")
            )
            DJEMBES_best_bid = (
                max(order_depths[Product.DJEMBES].buy_orders.keys())
                if order_depths[Product.DJEMBES].buy_orders
                else 0
            )
            DJEMBES_best_ask = (
                min(order_depths[Product.DJEMBES].sell_orders.keys())
                if order_depths[Product.DJEMBES].sell_orders
                else float("inf")
            )

            # Calculate the implied bid and ask for the synthetic basket
            implied_bid = (
                CROISSANTS_best_bid * CROISSANTS_PER_BASKET
                + JAMS_best_bid * JAMS_PER_BASKET
                + DJEMBES_best_bid * DJEMBES_PER_BASKET
            )
            implied_ask = (
                CROISSANTS_best_ask * CROISSANTS_PER_BASKET
                + JAMS_best_ask * JAMS_PER_BASKET
                + DJEMBES_best_ask * DJEMBES_PER_BASKET
            )

            # Calculate the maximum number of synthetic baskets available at the implied bid and ask
            if implied_bid > 0:
                CROISSANTS_bid_volume = (
                    order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                    // CROISSANTS_PER_BASKET
                )
                JAMS_bid_volume = (
                    order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                    // JAMS_PER_BASKET
                )
                if DJEMBES_PER_BASKET > 0:
                    DJEMBES_bid_volume = (
                        order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                        // DJEMBES_PER_BASKET
                    )
                else:
                    DJEMBES_bid_volume = float("inf")
                implied_bid_volume = min(
                    CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
                )
                synthetic_order_price[product].buy_orders[implied_bid] = implied_bid_volume

            if implied_ask < float("inf"):
                CROISSANTS_ask_volume = (
                    -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                    // CROISSANTS_PER_BASKET
                )
                JAMS_ask_volume = (
                    -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                    // JAMS_PER_BASKET
                )
                if DJEMBES_PER_BASKET > 0:
                    DJEMBES_ask_volume = (
                        -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                        // DJEMBES_PER_BASKET
                    )
                else:   
                    DJEMBES_ask_volume = float("inf")
                implied_ask_volume = min(
                    CROISSANTS_ask_volume, JAMS_ask_volume, DJEMBES_ask_volume
                )
                synthetic_order_price[product].sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def execute_spread_orders_old(
        self,
        order_depths: Dict[str, OrderDepth],
        current_positions: Dict[Product, int]
    ) -> Dict[str, List[Order]]:
        basket_analysis = []
        
        # Analyze both basket pairs
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths)
        for basket_product, synthetic_product in [
            (Product.PICNIC_BASKET1, Product.SYNTHETIC1),
            (Product.PICNIC_BASKET2, Product.SYNTHETIC2)
        ]:
            # Get order depths for both instruments
            basket_depth = order_depths.get(basket_product)
            synthetic_depth_prod = synthetic_depth.get(synthetic_product)
            
            if not basket_depth or not synthetic_depth_prod:
                continue
                
            # Calculate mid prices
            basket_mid = (min(basket_depth.sell_orders) + max(basket_depth.buy_orders)) / 2
            synthetic_mid = (min(synthetic_depth_prod.sell_orders) + max(synthetic_depth_prod.buy_orders)) / 2
            spread = basket_mid - synthetic_mid
            
            # Calculate position capacity
            position = current_positions.get(basket_product, 0)
            max_position = self.LIMIT[basket_product]
            remaining_capacity = max_position - abs(position)
            
            # Calculate potential profit
            profit_potential = abs(spread) * remaining_capacity
            
            basket_analysis.append({
                'basket': basket_product,
                'synthetic': synthetic_product,
                'spread_product': Product.SPREAD1 if basket_product == Product.PICNIC_BASKET1 else Product.SPREAD2,
                'spread': spread,
                'profit_potential': profit_potential,
                'position': position
            })

        if not basket_analysis:
            return {}

        # Sort by maximum profit potential
        sorted_baskets = sorted(basket_analysis, key=lambda x: x['profit_potential'], reverse=True)
        best_opportunity = sorted_baskets[0]
        #print("bestop",best_opportunity)
        if abs(best_opportunity['spread']) < self.params[best_opportunity['spread_product']]["default_spread_mean"]:
            return None
        # Determine trade direction
        target_position = self.params[best_opportunity['spread_product']]["target_position"]
        if best_opportunity['spread'] > 0:
            # Sell basket, buy synthetic
            basket_side = 'SELL'
            synthetic_side = 'BUY'
            target_position = -target_position
        else:
            # Buy basket, sell synthetic
            basket_side = 'BUY'
            synthetic_side = 'SELL'
            target_position = abs(target_position)
        
        # Calculate executable quantity
        current_basket_pos = best_opportunity['position']
        quantity = abs(target_position - current_basket_pos)
        
        # Get best available prices
        basket_orders = []
        synthetic_orders = []
        
        if basket_side == 'BUY':
            best_basket_price = max(order_depths[best_opportunity['basket']].buy_orders.keys())
            basket_orders.append(Order(best_opportunity['basket'], best_basket_price, quantity))
            
            best_synthetic_price = min(synthetic_depth[best_opportunity['synthetic']].sell_orders.keys())
            synthetic_orders.append(Order(best_opportunity['synthetic'], best_synthetic_price, -quantity))
        else:
            best_basket_price = min(order_depths[best_opportunity['basket']].sell_orders.keys())
            basket_orders.append(Order(best_opportunity['basket'], best_basket_price, -quantity))
            
            best_synthetic_price = max(synthetic_depth[best_opportunity['synthetic']].buy_orders.keys())
            synthetic_orders.append(Order(best_opportunity['synthetic'], best_synthetic_price, quantity))
        
        # Convert synthetic orders to component orders
        component_orders = self.convert_synthetic_basket_orders(
            synthetic_orders, 
            order_depths,
            best_opportunity['synthetic']
        )
        
        # Aggregate all orders
        return {
            best_opportunity['basket']: basket_orders,
            **component_orders
        }

    def execute_spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        current_positions: Dict[Product, int]
    ) -> Dict[str, List[Order]]:
        basket_analysis = []

        # Analyze both basket pairs
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths)

        for basket_product, synthetic_product in [
            (Product.PICNIC_BASKET1, Product.SYNTHETIC1),
            (Product.PICNIC_BASKET2, Product.SYNTHETIC2)
        ]:
            # Get order depths for both instruments
            basket_depth = order_depths.get(basket_product)
            synthetic_depth_prod = synthetic_depth.get(synthetic_product)

            if not basket_depth or not synthetic_depth_prod:
                continue

            # Calculate mid prices
            basket_mid = (min(basket_depth.sell_orders) + max(basket_depth.buy_orders)) / 2
            synthetic_mid = (min(synthetic_depth_prod.sell_orders) + max(synthetic_depth_prod.buy_orders)) / 2
            spread = basket_mid - synthetic_mid

            # Calculate position capacity
            position = current_positions.get(basket_product, 0)
            max_position = self.LIMIT[basket_product]
            remaining_capacity = max_position - abs(position)

            # Calculate potential profit
            profit_potential = abs(spread) * remaining_capacity

            basket_analysis.append({
                'basket': basket_product,
                'synthetic': synthetic_product,
                'spread_product': Product.SPREAD1 if basket_product == Product.PICNIC_BASKET1 else Product.SPREAD2,
                'spread': spread,
                'profit_potential': profit_potential,
                'position': position
            })

        if not basket_analysis:
            return {}

        orders_to_execute = {}

        for opportunity in basket_analysis:
            spread_threshold = self.params[opportunity['spread_product']]["default_spread_mean"]
            #print(opportunity['spread'] , spread_threshold, "diff spread")
            if abs(opportunity['spread']) < spread_threshold:
                continue

            # Determine trade direction
            target_position = self.params[opportunity['spread_product']]["target_position"]

            if opportunity['spread'] > 0:
                # Sell basket, buy synthetic
                basket_side = 'SELL'
                synthetic_side = 'BUY'
                target_position = -target_position
            else:
                # Buy basket, sell synthetic
                basket_side = 'BUY'
                synthetic_side = 'SELL'
                target_position = abs(target_position)

            # Calculate executable quantity
            current_basket_pos = opportunity['position']
            quantity = abs(target_position - current_basket_pos)

            # Get best available prices
            basket_orders = []
            synthetic_orders = []

            if basket_side == 'BUY':
                best_basket_price = max(order_depths[opportunity['basket']].buy_orders.keys())
                basket_orders.append(Order(opportunity['basket'], best_basket_price, quantity))

                best_synthetic_price = min(synthetic_depth[opportunity['synthetic']].sell_orders.keys())
                synthetic_orders.append(Order(opportunity['synthetic'], best_synthetic_price, quantity))
            else:
                best_basket_price = min(order_depths[opportunity['basket']].sell_orders.keys())
                basket_orders.append(Order(opportunity['basket'], best_basket_price, -quantity))

                best_synthetic_price = max(synthetic_depth[opportunity['synthetic']].buy_orders.keys())
                synthetic_orders.append(Order(opportunity['synthetic'], best_synthetic_price, -quantity))

            # Convert synthetic orders to component orders
            component_orders = self.convert_synthetic_basket_orders(
                synthetic_orders,
                order_depths,
                opportunity['synthetic']
            )

            # Aggregate all orders for this spread
            orders_to_execute[opportunity['basket']] = basket_orders
            for product in component_orders:
                if product not in orders_to_execute:
                    orders_to_execute[product] = []
                orders_to_execute[product].extend(component_orders[product])

        return orders_to_execute

    def convert_synthetic_basket_orders(
        self, 
        synthetic_orders: List[Order], 
        order_depths: Dict[str, OrderDepth],
        synthetic_type: Product
    ) -> Dict[str, List[Order]]:
        component_orders = {prod: [] for prod in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]}
        
        # Select appropriate weights
        weights = BASKET_WEIGHTS_1 if synthetic_type == Product.SYNTHETIC1 else BASKET_WEIGHTS_2
        
        for order in synthetic_orders:
            # Calculate component quantities
            croissant_qty = order.quantity * weights[Product.CROISSANTS]
            jams_qty = order.quantity * weights[Product.JAMS]
            DJEMBES_qty = order.quantity * weights.get(Product.DJEMBES, 0)
            
            # Get best available prices
            if order.quantity > 0:  # Buying synthetic = selling components
                croissant_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys()) if Product.DJEMBES in weights else 0
            else:  # Selling synthetic = buying components
                croissant_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                jams_price = min(order_depths[Product.JAMS].sell_orders.keys())
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys()) if Product.DJEMBES in weights else 0
            
            # Create component orders
            component_orders[Product.CROISSANTS].append(Order(
                Product.CROISSANTS, croissant_price, -croissant_qty if order.quantity > 0 else croissant_qty
            ))
            component_orders[Product.JAMS].append(Order(
                Product.JAMS, jams_price, -jams_qty if order.quantity > 0 else jams_qty
            ))
            if DJEMBES_qty != 0:
                component_orders[Product.DJEMBES].append(Order(
                    Product.DJEMBES, DJEMBES_price, -DJEMBES_qty if order.quantity > 0 else DJEMBES_qty
                ))
        
        return component_orders
        
    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        baskets: List[Product],
        basket_positions: List[int],
        spread_data: List[Dict[str, Any]],
    ):
        spread_signals = []
        
        # Analyze both spreads
        for i, (basket_product, spread_product) in enumerate([
            (Product.PICNIC_BASKET1, Product.SPREAD1),
            (Product.PICNIC_BASKET2, Product.SPREAD2)
        ]):
            if basket_product not in order_depths:
                continue
                
            # Calculate spread statistics
            basket_depth = order_depths[basket_product]
            synthetic_depth = self.get_synthetic_basket_order_depth(order_depths)[
                Product.SYNTHETIC1 if i == 0 else Product.SYNTHETIC2
            ]
            
            basket_swmid = self.get_swmid(basket_depth)
            synthetic_swmid = self.get_swmid(synthetic_depth)
            spread = basket_swmid - synthetic_swmid
            
            # Update spread history
            data = spread_data[i]
            data["spread_history"].append(spread)
            if len(data["spread_history"]) > self.params[spread_product]["spread_std_window"]:
                data["spread_history"].pop(0)
                
            # Calculate z-score if enough data
            if len(data["spread_history"]) >= self.params[spread_product]["spread_std_window"]:
                spread_std = np.std(data["spread_history"])
                zscore = (spread - self.params[spread_product]["default_spread_mean"]) / spread_std
                data["prev_zscore"] = zscore
                
                # Calculate potential return
                spread_value = abs(zscore) * spread_std
                position_capacity = self.params[spread_product]["target_position"] - abs(basket_positions[i])
                potential_return = spread_value * position_capacity
                
                spread_signals.append({
                    "index": i,
                    "zscore": abs(zscore),
                    "return": potential_return,
                    "direction": np.sign(zscore)
                })

        # Execute best opportunity
        if spread_signals:
            # Sort by potential return then z-score
            best_spread = max(spread_signals, key=lambda x: (x["return"], x["zscore"]))
            
            i = best_spread["index"]
            basket_product = baskets[i]
            spread_product = Product.SPREAD1 if i == 0 else Product.SPREAD2
            target_position = self.params[spread_product]["target_position"] * best_spread["direction"]
            
            return self.execute_spread_orders(
                target_position,
                basket_positions[i],
                order_depths,
                basket_product,
                Product.SYNTHETIC1 if i == 0 else Product.SYNTHETIC2
            )
            
        return None

    def classify_option_by_delta(self, delta):
        """Classify options based on delta values"""
        if abs(delta) > 0.995:  # Deep ITM
            return "ultra_hedge"  # Treat almost like underlying
        elif abs(delta) > 0.8:  # Strong ITM
            return "strong_hedge"
        elif abs(delta) > 0.6:  # Moderate ITM
            return "moderate_hedge"
        else:  # OTM or slight ITM
            return "option"  # Treat as a regular option

    def calculate_hedge_efficiency(self, product, order_depth, delta, state, volcanic_rock_mid_price):
        """Calculate the efficiency of using a particular instrument for hedging"""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            spread_percentage = spread / ((best_ask + best_bid) / 2)
            
            # Calculate transaction cost per unit delta
            if delta > 0:
                cost_per_delta = best_ask / delta  # Cost to buy per unit delta
            else:
                cost_per_delta = best_bid / abs(delta)  # Cost to sell per unit delta
                
            # Calculate liquidity score based on order book depth
            bid_volume = sum(order_depth.buy_orders.values())
            ask_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            liquidity_score = (bid_volume + ask_volume) / 2
            
            # Final efficiency score (lower is better)
            efficiency_score = cost_per_delta * (1 + spread_percentage * 2)
            
            return {
                'product': product,
                'delta': delta,
                'cost_per_delta': cost_per_delta,
                'spread_percentage': spread_percentage,
                'liquidity_score': liquidity_score,
                'efficiency_score': efficiency_score
            }
        return None

    def get_voucher_mid_price(self, voucher_order_depth, trader_data):
        """Calculate the mid price for a voucher product"""
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Store the price for future reference
            trader_data["prev_voucher_price"] = mid_price
            return mid_price
        elif trader_data["prev_voucher_price"] > 0:
            # Return previous price if no current market
            return trader_data["prev_voucher_price"]
        else:
            # Default fallback
            return 0
            
    def generate_delta_neutral_orders(self, state, volcanic_rock_mid_price, volcanic_rock_order_depth, total_delta_exposure=None):
        """Generate orders to maintain delta neutrality while respecting position limits"""
        # Calculate current delta exposure if not provided
        if total_delta_exposure is None:
            total_delta_exposure = self.calculate_total_delta_exposure(state, volcanic_rock_mid_price)
        
        # Calculate required hedge position
        target_hedge_position = -round(total_delta_exposure)
        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        print(target_hedge_position,current_position, "target hedge position")
        position_difference = target_hedge_position - current_position
        
        # Check if we need to adjust our position
        if abs(position_difference) < 30:
            return [] # Already delta neutral
        
        # Check position limits
        max_position = self.LIMIT[Product.VOLCANIC_ROCK]
        if abs(target_hedge_position) > max_position:
            # Scale back to respect position limits
            target_hedge_position = max_position * (1 if target_hedge_position > 0 else -1)
            position_difference = target_hedge_position - current_position
        
        # Generate orders to achieve target hedge position
        orders = []
        if position_difference > 0: # Need to buy
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(position_difference, abs(volcanic_rock_order_depth.sell_orders[best_ask]))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif position_difference < 0: # Need to sell
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(position_difference), volcanic_rock_order_depth.buy_orders[best_bid])
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
        
        return orders

    def optimize_delta_hedging(self, state, volcanic_rock_mid_price, total_delta_exposure):
        """Select the optimal combination of instruments for delta hedging"""
        hedging_candidates = []
        
        # Add underlying as a candidate
        if Product.VOLCANIC_ROCK in state.order_depths:
            hedging_candidates.append({
                'product': Product.VOLCANIC_ROCK,
                'delta': 1.0,  # Underlying always has delta of 1
                'order_depth': state.order_depths[Product.VOLCANIC_ROCK],
                'classification': 'ultra_hedge'  # Treat as ultra hedge
            })
        
        # Add all vouchers as candidates
        tte = TIME_TO_EXPIRY - (state.timestamp) / 1000000 / 252
        for voucher in VOLCANIC_VOUCHERS:
            if voucher in state.order_depths:
                if voucher in self.params and self.params[voucher].get("current_volatility"):
                    volatility = self.params[voucher]["current_volatility"]
                    strike = self.params[voucher]["strike"]
                    
                    # Calculate delta
                    delta = BlackScholesGreeks.delta(
                        volcanic_rock_mid_price,
                        strike,
                        tte,
                        volatility
                    )
                    
                    # Classify the option
                    classification = self.classify_option_by_delta(delta)
                    print(voucher, delta, "class", classification)
                    hedging_candidates.append({
                        'product': voucher,
                        'delta': delta,
                        'order_depth': state.order_depths[voucher],
                        'classification': classification
                    })
        
        # Calculate efficiency scores for all candidates
        scored_candidates = []
        for candidate in hedging_candidates:
            efficiency = self.calculate_hedge_efficiency(
                candidate['product'],
                candidate['order_depth'],
                candidate['delta'],
                state,
                volcanic_rock_mid_price
            )
            if efficiency:
                scored_candidates.append({**candidate, **efficiency})
        
        # Sort by efficiency (best hedging instruments first)
        scored_candidates.sort(key=lambda x: x['efficiency_score'])
        
        # Generate optimal hedging orders
        return self.generate_optimal_hedging_orders(scored_candidates, total_delta_exposure, state)

    def generate_optimal_hedging_orders(self, scored_candidates, total_delta_exposure, state):
        """Generate orders using the most efficient hedging instruments"""
        hedging_orders = {}
        remaining_delta = -total_delta_exposure  # Delta we need to hedge
        if np.abs(remaining_delta) < 30:
            return hedging_orders
        # First prioritize "ultra_hedge" options (delta > 0.95)
        ultra_hedge_candidates = [c for c in scored_candidates 
                                if c.get('classification') == 'ultra_hedge']
        
        # Then use other candidates
        other_candidates = [c for c in scored_candidates 
                        if c.get('classification') != 'ultra_hedge']
        
        # Combine, with ultra_hedge first
        prioritized_candidates = ultra_hedge_candidates + other_candidates
        #print(ultra_hedge_candidates, "ultra hedge candidates")
        for candidate in ultra_hedge_candidates:
            product = candidate['product']
            delta = candidate['delta']
            order_depth = candidate['order_depth']
            
            # Skip if no liquidity
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue
                
            # Calculate how much we can hedge with this instrument
            position_limit = self.LIMIT[product]
            current_position = state.position.get(product, 0)
            
            if remaining_delta > 0:  # Need positive delta exposure
                # We need to buy
                best_ask = min(order_depth.sell_orders.keys())
                available_volume = -order_depth.sell_orders[best_ask]
                
                # How many units can we buy?
                max_buy_quantity = min(
                    available_volume,
                    position_limit - current_position
                )
                
                # How much delta would this give us?
                delta_acquired = max_buy_quantity * delta
                
                # Limit to what we need
                if delta_acquired > remaining_delta:
                    adjusted_quantity = int(remaining_delta / delta)
                    quantity_to_buy = min(adjusted_quantity, max_buy_quantity)
                else:
                    quantity_to_buy = max_buy_quantity
                    
                if quantity_to_buy > 0:
                    if product not in hedging_orders:
                        hedging_orders[product] = []
                    hedging_orders[product].append(Order(product, best_ask, quantity_to_buy))
                    remaining_delta -= quantity_to_buy * delta
                    
            else:  # Need negative delta exposure
                # We need to sell
                best_bid = max(order_depth.buy_orders.keys())
                available_volume = order_depth.buy_orders[best_bid]
                
                # How many units can we sell?
                max_sell_quantity = min(
                    available_volume,
                    position_limit + current_position
                )
                
                # How much delta would this remove?
                delta_removed = max_sell_quantity * delta
                
                # Limit to what we need
                if delta_removed > abs(remaining_delta):
                    adjusted_quantity = int(abs(remaining_delta) / delta)
                    quantity_to_sell = min(adjusted_quantity, max_sell_quantity)
                else:
                    quantity_to_sell = max_sell_quantity
                    
                if quantity_to_sell > 0:
                    if product not in hedging_orders:
                        hedging_orders[product] = []
                    hedging_orders[product].append(Order(product, best_bid, -quantity_to_sell))
                    remaining_delta += quantity_to_sell * delta
            
            # If we've hedged enough, stop
            if abs(remaining_delta) < 0.1:
                break
                
        return hedging_orders

    def volcanic_voucher_orders(self, voucher, voucher_order_depth, voucher_position, trader_data, volatility, delta, total_delta_exposure, theoretical_value):
        """Generate orders for volcanic rock vouchers with delta neutrality in mind"""
        take_orders = []
        make_orders = []
        
        # Get position limit
        position_limit = self.LIMIT[voucher]
        
        # Calculate how much room we have for additional delta exposure
        remaining_delta_capacity = self.LIMIT[Product.VOLCANIC_ROCK] - abs(total_delta_exposure)
        
        # Adjust our trading based on delta capacity
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            
            
            # Store the theoretical price for future reference
            trader_data["last_theoretical_price"] = theoretical_value
            
            # Determine spread based on volatility
            
            # Adjust quantities based on delta impact
            max_buy_quantity = min(
                position_limit - voucher_position,
                int(remaining_delta_capacity / abs(delta)) if delta != 0 else position_limit
            )
            
            max_sell_quantity = min(
                position_limit + voucher_position,
                int(remaining_delta_capacity / abs(delta)) if delta != 0 else position_limit
            )

            print(theoretical_value, "theo", best_ask, best_bid, "best bid ask")
            buy_order_qty = 0
            sell_order_qty = 0
            # Take mispriced orders with adjusted quantities
            if best_bid > theoretical_value + self.params[voucher]["take_width"] and max_sell_quantity > 0:
                quantity = min(
                    voucher_order_depth.buy_orders[best_bid],
                    max_sell_quantity
                )
                if quantity > 0:
                    take_orders.append(Order(voucher, best_bid, -quantity))
                sell_order_qty += abs(quantity)
                    
            if best_ask < theoretical_value - self.params[voucher]["take_width"] and max_buy_quantity > 0:
                quantity = min(
                    -voucher_order_depth.sell_orders[best_ask],
                    max_buy_quantity
                )  
                if quantity > 0:
                    take_orders.append(Order(voucher, best_ask, quantity))
                buy_order_qty += abs(quantity)
            
            make_orders, _, _ = self.make_orders(
                voucher,
                voucher_order_depth,
                theoretical_value,
                voucher_position,
                buy_order_qty,
                sell_order_qty,
                self.params[voucher]["disregard_edge"],
                self.params[voucher]["join_edge"],
                self.params[voucher]["default_edge"],
                False,
                self.params[voucher].get("soft_position_limit", 0),
                )
        
        return take_orders, make_orders

    def calculate_total_delta_exposure(self, state, volcanic_rock_mid_price, pending_orders=None):
        """Calculate the total delta exposure across all voucher positions including pending orders"""
        total_delta = 0
        
        # Initialize dictionary to track pending order quantities
        pending_quantities = {voucher: 0 for voucher in [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]}
        
        # Add pending order quantities if provided
        if pending_orders:
            for product, orders in pending_orders.items():
                if product in pending_quantities:
                    for order in orders:
                        pending_quantities[product] += order.quantity
        
        # Calculate delta for each voucher including both positions and pending orders
        for voucher in pending_quantities.keys():
            if voucher in self.params:
                # Get current position (or 0 if no position)
                position = state.position.get(voucher, 0)
                
                # Add pending order quantity to position
                total_position = position + pending_quantities[voucher]
                
                # Only calculate delta if there's a position or pending order
                if total_position != 0:
                    tte = (
                        TIME_TO_EXPIRY
                        - (state.timestamp) / 1000000 / 252
                    )
                    volatility = self.params[voucher].get("current_volatility")
                    delta = BlackScholesGreeks.delta(
                        volcanic_rock_mid_price,
                        self.params[voucher]["strike"],
                        tte,
                        volatility
                    )
                    total_delta += delta * total_position
        
        
        return total_delta


    def calculate_implied_volatility(self, option_price, spot_price, strike, tte, voucher, trader_data):
        """Calculate implied volatility and maintain a rolling window"""
        # Calculate the current implied volatility using Newton-Raphson method
        try:
            if option_price <= spot_price-strike:
                print("toolow",option_price, spot_price, strike)
                raise ValueError("Option price is too low")
            current_iv = BlackScholesGreeks.implied_volatility(
                option_price, 
                spot_price, 
                strike, 
                tte
            )
            print("Current IV: for voucher ", voucher, current_iv)
        except:
            # Use previous value if calculation fails
            current_iv = trader_data[voucher].get("last_iv", 0.16)
        
        # Store the calculated IV in history
        if not np.isnan(current_iv):
            trader_data[voucher]["iv_history"].append(current_iv)
            # Keep only the last 20 values
            if len(trader_data[voucher]["iv_history"]) > 20:
                trader_data[voucher]["iv_history"].pop(0)
        
        # Calculate the rolling median if we have enough data points
        if len(trader_data[voucher]["iv_history"])== 20:
            rolling_iv = np.median(trader_data[voucher]["iv_history"])
        else:
            rolling_iv = current_iv
        
        # Store the last calculated IV for future reference
        trader_data[voucher]["last_iv"] = current_iv
        
        return rolling_iv
    def update_active_vouchers(self, volcanic_rock_mid_price):
        """Dynamically update which vouchers to trade based on current price"""
        active_vouchers = []
        
        # Always include the ATM and nearest OTM options
        atm_strike = round(volcanic_rock_mid_price / 250) * 250  # Round to nearest 250
        print("atm_strike" , atm_strike)
        # Add all ITM options as potential hedging instruments
        for strike in [9500, 9750, 10000, 10250, 10500]:
            if strike <= atm_strike - 700:  # Deep ITM
                continue
            elif abs(strike - volcanic_rock_mid_price) < 700:  # Near ATM
                active_vouchers.append(f"VOLCANIC_ROCK_VOUCHER_{strike}")
            elif strike > volcanic_rock_mid_price + 1000:  # Only include one OTM option
                active_vouchers.append(f"VOLCANIC_ROCK_VOUCHER_{strike}")
                break
                
        return active_vouchers

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            best_bid = max(state.order_depths[Product.SQUID_INK].buy_orders.keys())
            best_ask = min(state.order_depths[Product.SQUID_INK].sell_orders.keys())
            self.ink_history.append((best_bid + best_ask)/2)
            SQUID_INK_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            SQUID_INK_fair_value = self.SQUID_INK_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            SQUID_INK_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    SQUID_INK_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            SQUID_INK_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    SQUID_INK_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    SQUID_INK_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_INK_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                SQUID_INK_fair_value,
                SQUID_INK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
                True,
                self.params[Product.SQUID_INK]["soft_position_limit"],
            )
            result[Product.SQUID_INK] = (
                SQUID_INK_make_orders + SQUID_INK_take_orders + SQUID_INK_clear_orders
            )
        conversions = 1

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
        
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        current_positions = {
                Product.PICNIC_BASKET1: state.position.get(Product.PICNIC_BASKET1, 0),
                Product.PICNIC_BASKET2: state.position.get(Product.PICNIC_BASKET2, 0)
            }
            
            # Execute spread orders
        #spread_orders = self.execute_spread_orders(state.order_depths, current_positions)
        #if spread_orders:
        #    for product in spread_orders:
        #        result[product] = spread_orders[product]
        

        ##VOLCANO 
        # Process all volcanic rock vouchers
        if Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            
            volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            volcanic_rock_mid_price = (
                max(volcanic_rock_order_depth.buy_orders.keys()) +
                min(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            
            # Track pending orders for delta calculation
            pending_orders = {}
            
            # Calculate initial delta exposure without pending orders
            total_delta_exposure = self.calculate_total_delta_exposure(state, volcanic_rock_mid_price)
            print("init delta exposure", total_delta_exposure + volcanic_rock_position)
            valid_vouchers = self.update_active_vouchers(volcanic_rock_mid_price)
            print("valid vouchers", valid_vouchers)
            # Process each voucher product
            for voucher in valid_vouchers:
                if voucher in self.params and voucher in state.order_depths:
                    voucher_position = (
                        state.position[voucher]
                        if voucher in state.position
                        else 0
                    )
                    
                    # Initialize trader object if needed
                    if voucher not in traderObject:
                        traderObject[voucher] = {
                            "prev_voucher_price": 0,
                            "iv_history": [],
                            "last_iv": self.params[voucher].get("ivolatility"),
                            "last_theoretical_price": 0
                        }
                    
                    voucher_order_depth = state.order_depths[voucher]
                    voucher_mid_price = self.get_voucher_mid_price(
                        voucher_order_depth, 
                        traderObject[voucher]
                    )
                    
                    # Calculate time to expiry
                    tte = (
                        TIME_TO_EXPIRY
                        - (state.timestamp) / 1000000 / 252
                    )
                    
                    # Calculate implied volatility using rolling median
                    volatility = self.calculate_implied_volatility(
                        voucher_mid_price,
                        volcanic_rock_mid_price,
                        self.params[voucher]["strike"],
                        tte,
                        voucher,
                        traderObject
                    )
                    
                    # Store current volatility for future reference
                    self.params[voucher]["current_volatility"] = volatility
                    
                    # Calculate delta
                    delta = BlackScholesGreeks.delta(
                        volcanic_rock_mid_price,
                        self.params[voucher]["strike"],
                        tte,
                        volatility
                    )

                    theoretical_value = BlackScholesGreeks.call_price(
                        volcanic_rock_mid_price,
                        self.params[voucher]["strike"],
                        tte,
                        volatility
                    )
                    
                    # Generate orders with delta neutrality in mind
                    voucher_take_orders, voucher_make_orders = self.volcanic_voucher_orders(
                        voucher,
                        voucher_order_depth,
                        voucher_position,
                        traderObject[voucher],
                        volatility,
                        delta,
                        total_delta_exposure,
                        theoretical_value
                    )
                    
                    # Update total delta exposure based on new orders
                    for order in voucher_take_orders:
                        total_delta_exposure += delta * order.quantity
                    
                    # Add orders to result
                    if voucher_take_orders or voucher_make_orders:
                        pending_orders[voucher] = voucher_take_orders
                        print("voucher orders", voucher_take_orders, "make", voucher_make_orders)
                        result[voucher] = voucher_take_orders + voucher_make_orders
            
            total_delta_exposure = self.calculate_total_delta_exposure(
                state, 
                volcanic_rock_mid_price, 
                pending_orders
            )
            
            # Generate hedge orders for VOLCANIC_ROCK based on updated delta
            # volcanic_rock_orders = self.generate_delta_neutral_orders(
            #     state,
            #     volcanic_rock_mid_price,
            #     volcanic_rock_order_depth,
            #     total_delta_exposure  # Pass the updated delta exposure
            # )
            hedging_orders = self.optimize_delta_hedging(
                state,
                volcanic_rock_mid_price,
                total_delta_exposure
            )

            # Update the result dictionary
            for product, orders in hedging_orders.items():
                if product not in result:
                    result[product] = []
                result[product].extend(orders)

            print("orders" , pending_orders, "hedge", hedging_orders, "delta", total_delta_exposure)
            #if volcanic_rock_orders:
            #    result[Product.VOLCANIC_ROCK] = volcanic_rock_orders

        traderData = jsonpickle.encode(traderObject)
        #logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


