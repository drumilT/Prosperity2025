from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from collections import deque
import pandas as pd
import numpy as np

class Trader:
    POSITION_LIMIT = 50

    def __init__(self):
        self.window_size = 5
        self.ink_history  = deque(maxlen=self.window_size)
        self.kelp_history = deque(maxlen=self.window_size)

    def run(self, state: TradingState):
        print("traderData:", state.traderData)
        print("Observations:", state.observations)

        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == "RAINFOREST_RESIN":
                result[product] = self.trade_resin(depth, pos)
            elif product == "SQUID_INK":
                result[product] = self.trade_ink(depth, pos, state)

        traderData = "SAMPLE"
        conversions = 0
        return result, conversions, traderData

    def trade_resin(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        BUY_THRESHOLD = 9998
        SELL_THRESHOLD = 10002
        orders: List[Order] = []

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price <= BUY_THRESHOLD:
                max_buy = min(-ask_volume, self.POSITION_LIMIT - current_position)
                if max_buy > 0:
                    orders.append(Order("RAINFOREST_RESIN", ask_price, max_buy))
                    current_position += max_buy

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price >= SELL_THRESHOLD:
                max_sell = min(bid_volume, self.POSITION_LIMIT + current_position)
                if max_sell > 0:
                    orders.append(Order("RAINFOREST_RESIN", bid_price, -max_sell))
                    current_position -= max_sell

        return orders

    def trade_ink(self,
                  order_depth: OrderDepth,
                  current_position: int,
                  state: TradingState
                 ) -> List[Order]:
        orders: List[Order] = []

        features = self.get_nn_inputs(state)
        if features == None:
          pass
        score = self.neural_network(features)
        print(f"NN score for SQUID_INK: {score:.4f}")

        if score > 0.5:
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                volume = -ask_volume
                to_buy = min(volume, self.POSITION_LIMIT - current_position)
                if to_buy > 0:
                    orders.append(Order("SQUID_INK", ask_price, to_buy))
                    current_position += to_buy

        elif score < 0.5:
            for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                to_sell = min(bid_volume, self.POSITION_LIMIT + current_position)
                if to_sell > 0:
                    orders.append(Order("SQUID_INK", bid_price, -to_sell))
                    current_position -= to_sell

        return orders

    def get_nn_inputs(self, state: TradingState) -> np.ndarray | None:
        """
        Returns a 7×1 numpy column vector of features,
        or None if we don't yet have `window_size` periods of history.
        """
        def snapshot(depth: OrderDepth | None) -> np.ndarray:
            # If no depth provided, treat as empty
            buys  = sorted(depth.buy_orders.items(),  reverse=True) if depth else []
            sells = sorted(depth.sell_orders.items())             if depth else []
            bids  = [buys[i][1] if i < len(buys)  else 0 for i in range(3)]
            asks  = [-sells[i][1] if i < len(sells) else 0 for i in range(3)]
            best_bid = buys[0][0]  if buys  else 0
            best_ask = sells[0][0] if sells else 0
            mid = (best_bid + best_ask)/2 if buys and sells else 0
            return np.array([mid, *bids, *asks], dtype=float)

        # extract or None
        ink_depth  = state.order_depths.get("SQUID_INK")
        kelp_depth = state.order_depths.get("KELP")

        # append to history
        self.ink_history.append( snapshot(ink_depth) )
        self.kelp_history.append(snapshot(kelp_depth))

        # wait until full window
        if len(self.ink_history) < self.window_size:
            return None

        # stack to arrays
        ink_arr  = np.vstack(self.ink_history)   # shape (5,7)
        kelp_arr = np.vstack(self.kelp_history)

        # mid prices
        ink_mid  = ink_arr[:, 0]
        kelp_mid = kelp_arr[:, 0]

        # returns (pad front with zero)
        ink_ret  = np.concatenate(([0], np.diff(ink_mid)  / (ink_mid[:-1]  + 1e-6)))
        kelp_ret = np.concatenate(([0], np.diff(kelp_mid) / (kelp_mid[:-1] + 1e-6)))

        # skews
        ink_skew  = ink_arr[:, 1:4].sum(1) / (ink_arr[:, 4:7].sum(1) + 1e-6) - 1
        kelp_skew = kelp_arr[:, 1:4].sum(1) / (kelp_arr[:, 4:7].sum(1) + 1e-6) - 1

        # rolling momentum (simple convolution) & reversion
        kernel = np.ones(self.window_size)/self.window_size
        kelp_mom = np.convolve(kelp_ret, kernel, mode='valid')
        kelp_mom = np.concatenate((np.zeros(self.window_size-1), kelp_mom))
        kelp_rev = kelp_ret - kelp_mom

        ink_mom = np.convolve(ink_ret, kernel, mode='valid')
        ink_mom = np.concatenate((np.zeros(self.window_size-1), ink_mom))
        ink_rev = ink_ret - ink_mom

        # build final 7×1 feature vector from last row
        last = -1
        features = np.array([
            kelp_ret[last],
            ink_skew[last],
            kelp_skew[last],
            kelp_mom[last],
            kelp_rev[last],
            ink_mom[last],
            ink_rev[last],
        ], dtype=float)
        features = np.nan_to_num(features, nan=0.0)

        return features
    
    def neural_network(self, x):
        # Convert input to numpy array

        # Layer 1 (Linear + ReLU)
        W1 = np.array([
            [0.131016, -0.268745, 0.0563415, 0.191271, -0.0998127, 0.563222, 1.38176],
            [0.0726058, -0.210003, -0.187908, 0.0323092, 0.0535123, -0.502474, -0.584985],
            [0.209164, -0.272709, 0.205284, -0.0656244, -0.212869, -0.338386, -1.06647],
            [0.194315, 0.205377, -0.411796, 0.278795, 0.305473, 0.436267, 0.651472],
            [-0.256794, 0.0521263, -0.0481285, 0.125035, 0.232011, -0.617022, -1.44941],
            [-0.0528685, -0.156306, 0.236477, -0.0942886, -0.0528191, 0.572591, 1.16497],
            [0.0232669, -0.0203574, 0.0342667, 0.0376158, -0.0791764, -0.511974, -0.935861],
            [0.0947581, -0.13206, 0.117769, -0.0691683, -0.0402917, -0.611776, -1.61821],
            [0.0188643, 0.00700093, 0.016606, -0.0405633, 0.0228104, -0.39905, -1.26722],
            [-0.238979, -0.0728445, 0.105986, -0.0211354, 0.276913, 0.332867, 1.38899],
            [-0.0309747, 0.0242813, 0.00175691, 0.111248, 0.0521331, -0.400084, -0.734029],
            [0.278067, 0.141434, -0.130653, -0.10994, -0.167683, -0.365273, -1.1056],
            [-0.272707, 0.220285, -0.0672001, -0.0347397, 0.268775, -0.491551, -1.05941],
            [0.0107392, -0.254025, 0.213041, -0.0178641, -0.0520602, -0.556104, -1.51828],
            [-0.0444617, -0.274074, -0.554939, -0.090784, -0.170849, 0.200611, 1.3259],
            [-0.0586019, 0.21144, -0.14259, 0.0154754, 0.0995943, 0.434791, 1.69984],
            [0.321549, -0.0789964, -0.0150408, 0.123328, 0.0318962, -0.303311, -0.771301],
            [-0.109637, 0.380119, 0.128304, 0.156365, 0.143301, -0.191217, 0.255529],
            [-0.298004, -0.0928979, 0.0315088, 0.213732, 0.285867, 0.482017, 1.75111],
            [0.0329035, -0.264118, 0.253256, 0.0830713, 0.0697287, 0.470322, 1.32078],
            [-0.167647, 0.100316, 0.0523709, -0.0434838, 0.175209, 0.405919, 1.47503],
            [-0.187806, -0.0219601, 0.00578231, 0.0820971, 0.134621, -0.516354, -1.36601],
            [0.158988, 0.216703, 0.319183, 0.270933, 0.220619, -0.299126, -0.39401],
            [0.0083854, -0.0927915, 0.0684932, 0.00790601, -0.0296788, -0.704922, -1.90339],
            [-0.116428, 0.0987176, -0.0792177, 0.0251125, 0.111022, -0.70691, -1.9592],
            [0.107472, 0.401008, -0.285424, -0.0884848, -0.164258, 0.705981, 1.92432],
            [-0.0165724, -0.239405, -0.203151, -0.250033, 0.246864, 0.269177, 0.313693],
            [0.0901434, 0.230405, -0.19986, 0.00309982, -0.0761995, -0.832694, -2.32048],
            [0.00482351, -0.0267656, 0.0389246, -0.00643308, -0.014193, -0.574588, -1.38495],
            [-0.0181249, 0.0373112, -0.0452386, 0.0221074, 0.0316606, -0.834978, -2.42079],
            [0.0209875, -0.0785965, 0.0713331, 0.109714, -0.0531628, -0.100523, 0.187213],
            [0.0265042, -0.233422, -0.401467, -0.327719, 0.260334, 0.0258621, -0.148277]
        ])
        b1 = np.array([
            0.596815, 0.443866, 0.506646, 0.0744749, 0.61075, 0.380327, 0.560293, 0.608222,
            0.496941, 0.33111, 0.478664, 0.517478, 0.600501, 0.578781, 0.299811, 0.546336,
            0.397766, -0.209767, 0.43245, 0.559079, 0.54744, 0.514875, -0.0712899, 0.603795,
            0.569235, 0.373278, 0.0601779, 0.694508, 0.532884, 0.738661, -0.44603, -0.239238
        ])

        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)  # ReLU

        # Layer 2 (Linear + Sigmoid)
        W2 = np.array([[0.823313, -0.447696, -0.920856, 0.336113, -0.819381, 0.773148, -0.92906, -0.896847,
                        -0.877376, 0.920933, -0.352246, -0.715025, -0.859041, -0.977223, 0.569232, 0.861249,
                        -0.484176, 0.160393, 1.00203, 0.906712, 0.888419, -1.25179, -0.211565, -1.01044,
                        -1.13607, 0.93114, 0.182528, -1.10322, -1.2421, -1.43727, -0.00952682, -0.00523067]])
        b2 = np.array([0.0980446])

        z2 = np.dot(W2, a1) + b2
        a2 = 1 / (1 + np.exp(-z2))  # Sigmoid

        return float(a2[0])
            