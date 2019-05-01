import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import load_data
import json

def load_model():
    with open('svr_speed_model.pickle','rb') as f:
        svr_speed = pickle.load(f)

    with open('svr_flow_model.pickle','rb') as f:
        svr_flow = pickle.load(f)

    with open('lstm_speed.pickle','rb') as f:
        lstm_speed = pickle.load(f)

    with open('lstm_flow.pickle','rb') as f:
        lstm_flow = pickle.load(f)

    return svr_speed,svr_flow,lstm_speed,lstm_flow

if __name__ == '__main__':
    speedload_data('data/test_up.xlsx')
