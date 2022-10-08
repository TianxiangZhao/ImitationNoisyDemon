import os

import ipdb
import numpy as np
import random
import math


class meters:
    """
    collects results at each batch, used for testing
    params:
        orders: norms in calculation. update follows:
            ((a1^orders+a2^orders+...+ak^orders)/k)^(1/orders)
    """
    def __init__(self, orders=1):
        self.avg_value = 0
        self.tot_weight = 0
        self.orders = orders

    def update(self, value, weight=1.0):
        value = float(value)

        update_step = self.tot_weight/(self.tot_weight+weight)
        record = math.pow(self.avg_value, self.orders)*update_step + math.pow(value, self.orders)*(1-update_step)
        self.avg_value = math.pow(record, 1/self.orders)
        self.tot_weight += weight

    def avg(self):

        return self.avg_value
