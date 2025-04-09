from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    POSITION_LIMIT = 50
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}

        for product in state.order_depths:
            if product == "RAINFOREST_RESIN":
                result[product] = self.trade_resin(state.order_depths[product], state.position.get(product, 0))

        traderData = "SAMPLE"
        conversions = 0
        return result, conversions, traderData
    
    def trade_resin(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
      BUY_THRESHOLD = 9998
      SELL_THRESHOLD = 10002
      orders = []

      # BUY if ask price <= BUY_THRESHOLD and position < limit
      for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
          if ask_price <= BUY_THRESHOLD:
              max_buy_volume = min(-ask_volume, self.POSITION_LIMIT - current_position)
              if max_buy_volume > 0:
                  print(f"BUY {max_buy_volume}x @ {ask_price}")
                  orders.append(Order("RAINFOREST_RESIN", ask_price, max_buy_volume))
                  current_position += max_buy_volume
                  print("Yes")

      # SELL if bid price >= SELL_THRESHOLD and position > -limit
      for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
          if bid_price >= SELL_THRESHOLD:
              max_sell_volume = min(bid_volume, self.POSITION_LIMIT + current_position)
              if max_sell_volume > 0:
                  print(f"SELL {max_sell_volume}x @ {bid_price}")
                  orders.append(Order("RAINFOREST_RESIN", bid_price, -max_sell_volume))
                  current_position -= max_sell_volume
                  print("YES")

      return orders
