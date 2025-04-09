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
        if features is None:
          return []
        score = np.dot(self.neural_network(features), np.array([-1,0,1]))
        print(f"NN score for SQUID_INK: {score:.4f}")

        if score == 1:
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                volume = -ask_volume
                to_buy = min(volume, self.POSITION_LIMIT - current_position)
                if to_buy > 0:
                    orders.append(Order("SQUID_INK", ask_price, to_buy))
                    current_position += to_buy

        elif score == -1:
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
        W1 = np.array([[-2.1899e-01, -1.4161e-01,  1.2196e-02,  7.4485e-02,  2.4623e-01,
                       -9.3426e-01, -2.5621e+00,  1.6417e-01],
                      [ 1.5081e-01,  1.0998e-02,  4.7894e-02, -5.0100e-02, -1.5333e-01,
                        1.1008e+00,  2.9804e+00, -3.2919e-01],
                      [ 1.6089e-02, -9.4563e-02, -3.1321e-02,  2.9833e-02,  3.4037e-02,
                       -8.9530e-01, -2.6164e+00, -3.2681e-02],
                      [ 1.1350e-01,  3.2890e-02,  7.2802e-02, -5.3153e-02, -1.1042e-01,
                       -1.0532e+00, -2.7472e+00,  4.7673e-02],
                      [-3.2022e-01, -2.0464e-01,  1.1306e-01,  9.2071e-02,  3.7613e-01,
                        9.0398e-01,  2.2858e+00, -3.3352e-01],
                      [-9.6083e-02,  3.0896e-01, -1.8804e-01, -1.9781e-01, -4.4364e-01,
                        1.0834e-01,  2.6142e-01,  6.4615e-01],
                      [ 3.7074e-02, -2.5875e-01,  2.1833e-01, -3.7479e-01, -1.9685e-02,
                       -5.6169e-01, -1.3015e+00,  5.1457e-01],
                      [ 2.1912e-01,  4.8858e-02,  2.4482e-02, -1.5948e-01, -1.3764e-01,
                       -9.2870e-01, -2.4446e+00, -1.0592e-01],
                      [-7.2143e-02, -9.4854e-02,  1.3407e-01,  2.8583e-01,  4.2819e-03,
                       -5.7909e-01, -1.8297e+00,  5.1505e-01],
                      [ 1.1696e-01,  1.7547e-02,  5.3136e-02,  1.0347e-01, -1.8043e-01,
                       -6.5727e-01, -2.0548e+00, -2.1322e-01],
                      [ 2.3997e-01,  2.0179e-02,  6.4572e-02, -9.4060e-03, -2.6054e-01,
                       -1.0186e+00, -2.4705e+00,  1.0633e-01],
                      [-5.8158e-03, -1.4350e-01,  1.1739e-01,  9.0375e-03,  1.0895e-02,
                        1.1498e+00,  3.0647e+00, -2.8782e-01],
                      [ 2.3545e-01,  2.4794e-01, -1.9653e-01, -8.9112e-02, -7.7749e-02,
                       -8.7030e-01, -2.2158e+00,  3.9445e-01],
                      [-1.7965e-01, -1.2165e-01,  1.1923e-01,  9.7784e-02,  1.8665e-01,
                        8.2676e-01,  2.0230e+00, -6.1666e-01],
                      [ 2.3817e-02,  2.2428e-02, -1.5704e-03, -3.5931e-02,  2.4945e-02,
                       -7.5679e-01, -2.0418e+00,  4.0866e-01],
                      [ 5.9909e-02, -3.0830e-01,  2.7125e-01,  7.3657e-03, -3.8702e-02,
                        7.2640e-01,  1.9087e+00, -1.5173e+00],
                      [-2.2253e-02,  8.3578e-02,  1.6155e-02, -7.0830e-02,  5.6130e-02,
                        3.9709e-01,  1.1616e+00, -5.1973e-02],
                      [ 7.8110e-02, -2.4606e-01,  1.9914e-01, -1.2496e-01, -6.6213e-02,
                        8.3114e-01,  1.9503e+00,  7.5323e-02],
                      [ 2.1202e-02,  1.2651e-01,  6.3662e-02,  6.0660e-02,  6.3462e-02,
                        4.2239e-01,  1.3846e+00, -1.5655e+00],
                      [-3.7196e-01,  3.0210e-01, -2.8891e-01, -1.5188e-02,  4.1417e-01,
                        6.6299e-01,  1.7979e+00, -3.8604e-01],
                      [ 6.4803e-02,  7.8656e-02, -1.3667e-01, -1.1766e-01,  8.0116e-03,
                        1.0228e+00,  2.6533e+00, -1.2653e-01],
                      [-1.7098e-02, -2.2184e-01,  1.7681e-01, -1.0921e-01,  7.0099e-02,
                       -2.3633e-01, -1.0840e+00, -1.1115e-01],
                      [-2.8564e-01, -7.2102e-02,  3.8678e-02,  7.5926e-02,  2.6811e-01,
                        1.1379e+00,  3.0201e+00, -2.0537e-01],
                      [-7.1554e-02, -1.7920e-01,  7.0585e-02,  1.2379e-01,  9.8967e-02,
                        6.7062e-01,  1.6412e+00, -1.9189e-01],
                      [ 9.9031e-02,  6.7311e-03, -2.4643e-01, -9.3949e-02, -1.3169e-01,
                       -4.5180e-02,  2.3246e-01,  2.9744e-01],
                      [ 1.9824e-01,  2.1761e-01, -1.0120e-01, -7.5629e-02, -2.1644e-01,
                        9.8946e-01,  2.6560e+00, -2.1982e-01],
                      [-2.9126e-01, -3.9521e-01,  1.1353e-01,  2.7375e-01, -6.7613e-02,
                       -1.4957e-01, -3.8236e-01, -4.2235e-01],
                      [ 1.4033e-01, -1.1049e-02, -1.6528e-01, -7.3585e-02, -2.7674e-02,
                       -8.5151e-01, -2.5173e+00,  1.3811e-01],
                      [-4.8218e-02, -5.3758e-02, -7.8146e-02,  2.3536e-02,  6.8625e-02,
                       -9.6734e-01, -2.5684e+00,  2.0704e-01],
                      [ 8.2295e-02, -1.0257e-03, -1.1079e-01,  7.4676e-02, -7.7311e-02,
                        9.0635e-01,  2.5864e+00,  1.8387e-02],
                      [ 3.4591e-01,  4.6901e-02,  1.0798e-01,  8.6902e-02, -3.9164e-01,
                       -8.0261e-01, -1.6840e+00, -3.7194e-01],
                      [-1.3727e-01, -1.5904e-02, -2.0502e-01,  1.7185e-01,  2.7680e-01,
                        4.5139e-01,  1.5586e+00, -1.5528e+00]])
        b1 = np.array([-0.3360, -0.3762, -0.5166, -0.2915,  1.1421,  0.1763,  1.2605, -0.4041,
                       1.6732,  0.2437, -0.1635, -0.4909,  1.3019,  1.1215,  0.7457,  0.7015,
                       1.2742, -0.4250,  0.6499,  1.0676, -0.3576,  1.4885, -0.5665,  1.0721,
                       1.1582, -0.3790,  1.2171, -0.2236, -0.2820, -0.5942, -0.0906,  0.5251])

        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)  # ReLU

        # Layer 2 (Linear + Sigmoid)
        W2 = np.array([[ 1.5843, -0.1476,  1.0310,  1.9702, -1.4940, -0.5438,  0.1760,  1.4424,
                        0.2318,  0.5424,  1.2578, -0.2302,  0.6618, -1.6170,  0.8426, -0.9696,
                       -1.6230, -0.5040, -0.7837, -2.1785, -0.4867, -0.1057, -0.3857, -1.4028,
                       -0.9080, -0.1181, -0.3042,  1.2575,  1.6467, -0.3872,  0.6599, -0.7855],
                      [-2.4200, -2.8918, -1.9187, -2.7278,  0.0496,  0.1014,  0.6652, -2.0029,
                        0.6090, -0.7532, -2.0030, -3.5707,  0.7036,  0.4367,  0.1840,  1.4095,
                        0.5135, -0.9703,  0.6699,  0.3883, -1.2329,  0.6562, -2.6041,  0.0892,
                        0.3155, -1.7310,  0.4444, -1.5380, -2.5634, -1.7160, -0.8790,  1.2285],
                      [-0.0861,  1.9719, -0.0315, -0.4021,  0.4975,  0.1040, -1.2440, -0.3425,
                       -1.4858, -0.5818, -0.3330,  2.9587, -1.7472,  0.9287, -1.9696, -0.5266,
                        0.1264,  0.8696, -0.5115,  0.6094,  0.9820, -1.0476,  1.7868,  0.3821,
                       -0.2729,  1.1646, -0.7730, -0.3113, -0.0935,  1.2528, -0.3541, -0.9902]])
        b2 = np.array([-0.6928,  0.6123, -0.4830])

        z2 = np.dot(W2, a1) + b2
        return np.exp(z2) / sum(np.exp(z2))

            