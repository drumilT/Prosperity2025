from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
from collections import deque

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
        "default_spread_mean": 20,
        "default_spread_std": 85,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 20,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 100,
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
            Product.SYNTHETIC2: 100
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
                    print("Volatility: ", volatility)
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
        print("bestop",best_opportunity)
        if abs(best_opportunity['spread'] - self.params[best_opportunity['spread_product']]["default_spread_mean"]) < 0 :
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
                synthetic_orders.append(Order(opportunity['synthetic'], best_synthetic_price, -quantity))
            else:
                best_basket_price = min(order_depths[opportunity['basket']].sell_orders.keys())
                basket_orders.append(Order(opportunity['basket'], best_basket_price, -quantity))

                best_synthetic_price = max(synthetic_depth[opportunity['synthetic']].buy_orders.keys())
                synthetic_orders.append(Order(opportunity['synthetic'], best_synthetic_price, quantity))

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
        print("orders_to_execute", orders_to_execute)
        # Return all orders to execute
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
                Product.CROISSANTS, croissant_price, -croissant_qty 
            ))
            component_orders[Product.JAMS].append(Order(
                Product.JAMS, jams_price, -jams_qty
            ))
            if DJEMBES_qty != 0:
                component_orders[Product.DJEMBES].append(Order(
                    Product.DJEMBES, DJEMBES_price, -DJEMBES_qty
                ))
        
        return component_orders
    
    def convert_synthetic_basket_orders_new(self, synthetic_orders, order_depths, synthetic_type):
        component_orders = {prod: [] for prod in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]}

        # Select appropriate weights
        weights = BASKET_WEIGHTS_1 if synthetic_type == Product.SYNTHETIC1 else BASKET_WEIGHTS_2

        # Temporary storage for net quantities
        net_quantities = {Product.CROISSANTS: 0, Product.JAMS: 0, Product.DJEMBES: 0}

        for order in synthetic_orders:
            # Calculate component quantities
            croissant_qty = order.quantity * weights[Product.CROISSANTS]
            jams_qty = order.quantity * weights[Product.JAMS]
            DJEMBES_qty = order.quantity * weights.get(Product.DJEMBES, 0)

            # Update net quantities
            net_quantities[Product.CROISSANTS] += croissant_qty
            net_quantities[Product.JAMS] += jams_qty
            net_quantities[Product.DJEMBES] += DJEMBES_qty

        # Create orders only for net quantities that are non-zero
        for product, net_qty in net_quantities.items():
            if net_qty != 0:
                if net_qty > 0:  
                    price = max(order_depths[product].buy_orders.keys())
                else: 
                    price = min(order_depths[product].sell_orders.keys())

                component_orders[product].append(Order(
                    product, price, net_qty
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
        spread_orders = self.execute_spread_orders(state.order_depths, current_positions)
        if spread_orders:
            for product in spread_orders:
                result[product] = spread_orders[product]

        traderData = jsonpickle.encode(traderObject)
        #logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
