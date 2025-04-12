from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from collections import deque
import pandas as pd
import numpy as np

class Trader:
    POSITION_LIMIT = 50

    def __init__(self):
        self.window_size = 6
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
        if features is None:
          return []
        logits = self.neural_network(features)
        score = np.argmax(logits)-1
        print("NN score for SQUID_INK:", score)

        if int(score) == 1:
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                volume = -ask_volume
                to_buy = min(volume, self.POSITION_LIMIT - current_position)
                if to_buy > 0:
                    orders.append(Order("SQUID_INK", ask_price, to_buy))
                    current_position += to_buy

        elif int(score) == -1:
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
        ink_ret  =  np.diff(ink_mid)  / (ink_mid[:-1]  + 1e-6)
        kelp_ret =  np.diff(kelp_mid) / (kelp_mid[:-1] + 1e-6)

        # skews
        ink_skew  = ink_arr[:, 1:4].sum(1) / (ink_arr[:, 4:7].sum(1) + 1e-6) - 1
        kelp_skew = kelp_arr[:, 1:4].sum(1) / (kelp_arr[:, 4:7].sum(1) + 1e-6) - 1

        # rolling momentum (simple convolution) & reversion
        kernel = np.ones(self.window_size-1)/(self.window_size-1)
        kelp_mom = np.convolve(kelp_ret, kernel, mode='valid')
        kelp_mom = np.concatenate((np.zeros(self.window_size-2), kelp_mom))
        kelp_rev = kelp_ret - kelp_mom

        ink_mom = np.convolve(ink_ret, kernel, mode='valid')
        ink_mom = np.concatenate((np.zeros(self.window_size-2), ink_mom))
        ink_rev = ink_ret - ink_mom
 
        # build final 7×1 feature vector from last row
        last = -1
        #print("kelp_ret" , kelp_ret)
        #print("ink_ret" , ink_ret)
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
        print("input" ,x)
        # Layer 1 (Linear + ReLU)
        W1 = np.array([[-5.5969e-02, -2.0235e-02,  4.9472e-02, -5.5413e-02,  1.3491e-01,
                       -8.0153e-01, -2.1091e+00],
                      [ 4.1157e-03,  2.1348e-01, -3.7911e-01, -7.1412e-02,  9.7877e-02,
                        8.2513e-01,  2.1243e+00],
                      [-1.1316e-01, -1.9930e-01,  1.6997e-01,  6.2897e-02,  2.2904e-01,
                       -5.7487e-01, -1.4938e+00],
                      [-8.7163e-02, -6.5811e-02, -7.6227e-02,  2.0764e-02,  3.0992e-02,
                       -1.1336e+00, -3.0596e+00],
                      [ 3.5930e-01, -1.7833e-01,  2.4522e-01, -9.1097e-02, -2.6445e-01,
                        1.0283e+00,  2.6510e+00],
                      [-2.2661e-01,  9.2908e-02, -2.9289e-01,  7.3595e-02,  1.3665e-01,
                        6.3057e-01,  1.5029e+00],
                      [ 3.9368e-02, -3.3399e-01,  1.2952e-01,  4.7812e-02,  1.1469e-01,
                        5.7062e-01,  1.5529e+00],
                      [-1.7177e-01,  2.6109e-03, -7.5293e-02,  7.0613e-02,  1.5127e-01,
                       -6.9584e-01, -2.0309e+00],
                      [-2.2035e-01,  8.8233e-02, -2.0744e-02,  2.0915e-01,  9.2745e-02,
                       -2.9226e-01, -7.2724e-01],
                      [ 1.6380e-01,  1.8404e-02, -1.5534e-01,  2.9957e-02, -2.2651e-01,
                        1.0281e+00,  2.7123e+00],
                      [-8.8076e-02,  2.0080e-01, -4.9840e-02, -1.2897e-01, -2.4277e-01,
                        2.2565e-01, -5.2790e-02],
                      [-5.6681e-02, -9.9460e-02, -4.3755e-02,  1.9927e-02,  7.6113e-02,
                        1.1079e+00,  2.8709e+00],
                      [ 2.0475e-01,  1.4109e-01, -1.4593e-02, -7.2892e-02, -1.9701e-01,
                        1.0187e+00,  2.5923e+00],
                      [ 1.5901e-01, -3.7843e-01,  4.3846e-01,  2.9054e-01, -3.8707e-01,
                        1.2431e-01,  2.3638e-02],
                      [ 1.6475e-01, -9.9507e-02,  8.4225e-02,  2.3754e-01, -2.0982e-01,
                       -6.4438e-01, -1.1680e+00],
                      [ 3.7431e-01,  4.9249e-02,  8.8380e-02, -6.4162e-02, -2.9949e-01,
                       -1.0506e+00, -2.7322e+00],
                      [-2.7399e-01, -3.3764e-01,  3.8725e-02,  1.4581e-01, -2.7149e-01,
                       -1.9729e-01, -6.4473e-01],
                      [-1.2971e-01,  2.7216e-01,  2.8629e-02,  3.1704e-01,  1.0269e-01,
                        2.7020e-01,  1.2013e+00],
                      [ 9.6902e-02, -1.1603e-02,  1.7419e-02, -2.3686e-02, -8.0268e-02,
                       -7.8378e-01, -2.0845e+00],
                      [-5.3189e-02, -9.5340e-02, -3.4814e-02,  4.9939e-03, -5.8909e-04,
                       -1.0568e+00, -2.6882e+00],
                      [ 3.2782e-02, -1.3680e-02, -3.4035e-02,  5.2240e-03, -1.0897e-01,
                        8.7443e-01,  2.7270e+00],
                      [ 4.2387e-02, -9.6439e-02, -2.4982e-01, -1.0240e-01, -1.1047e-04,
                       -2.1772e-01, -9.2804e-01],
                      [ 3.5927e-01,  5.3489e-02,  7.8312e-02, -7.0569e-02, -2.7826e-01,
                       -8.4754e-01, -2.3052e+00],
                      [ 6.6661e-02, -1.5894e-01,  1.3766e-01, -1.1984e-01,  3.8539e-02,
                        6.1997e-01,  1.5857e+00],
                      [-8.3103e-02, -4.0407e-02, -8.7310e-02,  2.5488e-02,  4.1436e-02,
                       -1.0105e+00, -2.6948e+00],
                      [ 4.4525e-03, -2.9074e-01,  2.6619e-01,  3.8596e-02, -6.7655e-02,
                        5.9574e-01,  1.9737e+00],
                      [-1.3957e-01,  2.9068e-02, -5.0326e-02,  1.3634e-01,  8.0407e-02,
                       -4.6317e-01, -1.2587e+00],
                      [ 1.2558e-01,  2.6814e-01, -2.7578e-01, -7.5584e-02, -4.8182e-02,
                       -5.2294e-01, -1.6408e+00],
                      [-2.2272e-02,  1.0955e-02, -1.0410e-01,  2.1409e-02,  1.3122e-02,
                       -8.9816e-01, -2.4650e+00],
                      [ 3.1865e-01,  6.6820e-02,  9.2560e-02, -1.0049e-01, -2.3670e-01,
                       -9.8293e-01, -2.7740e+00],
                      [ 3.2986e-02, -3.3019e-01,  2.4032e-01, -2.2457e-01,  1.3746e-01,
                       -4.1972e-01, -9.3516e-01],
                      [ 3.2073e-01,  1.3730e-01,  1.3841e-01, -3.2212e-01,  3.6103e-01,
                        1.6182e-02, -1.9168e-01]])
        b1 = np.array([ 1.5383,  1.6736,  1.5749, -0.4377, -0.4344,  1.2201,  1.4005, -0.4142,
                       1.1672, -0.4402,  1.4728, -0.5536, -0.3779, -0.1923,  1.4111, -0.2700,
                       1.5462,  1.5769,  1.1712, -0.3072, -0.5059,  1.3581, -0.2457,  0.8496,
                      -0.3934, -0.3647,  0.9704,  1.1828, -0.4428, -0.3225,  1.2447,  0.8878])

        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)  # ReLU
        # Layer 2 (Linear + Sigmoid)
        W2 = np.array([[ 0.1175, -1.5759, -0.3367,  2.2504, -0.1932, -1.7152, -1.8849,  1.3653,
                       -0.4069, -0.7402, -0.8390, -0.2143, -0.3804,  0.0929, -0.2527,  1.9264,
                       -0.7546, -1.2709,  0.5049,  1.6649, -0.2570, -0.5681,  1.4329, -1.1025,
                        2.5519, -0.1135,  0.0128,  0.3134,  2.0341,  1.5223, -0.3810, -0.3752],
                      [ 0.9402,  0.0687,  0.8153, -2.7794, -2.6539,  0.2268,  0.3385, -1.3931,
                        0.3849, -2.4222,  0.4254, -2.5322, -2.8937,  0.3318,  0.5474, -2.7512,
                        0.6991,  0.2873,  0.6043, -2.7543, -1.5782,  0.6698, -2.4537, -0.3593,
                       -3.1466, -0.5968,  0.6177,  0.5952, -2.6925, -2.5523,  0.8173,  0.3323],
                      [-1.8890,  0.4850, -1.5520, -0.4994,  1.8556,  0.7094,  0.5761, -0.3983,
                       -0.7247,  1.7780, -0.3172,  1.6739,  1.9727,  0.2263, -1.1783, -0.3513,
                       -0.8519,  0.1213, -2.1519, -0.2806,  1.0845, -1.0536, -0.6352,  0.8418,
                       -0.2573,  0.7435, -1.2990, -1.3640, -0.2595, -0.0370, -1.0847, -0.3736]])
        b2 = np.array([-0.6049,  0.6190, -0.5414])

        z2 = np.dot(W2, a1) + b2
        
        softmax = np.exp(z2) / sum(np.exp(z2))
        return softmax

            