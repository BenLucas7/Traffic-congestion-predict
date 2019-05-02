import configparser
import pickle
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from utils import get_SVR_input
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def test():
    test_x, test_y,time, scalers = get_SVR_input('test_up','speed')


    # Load the prediction pipeline.
    with open('pickle/svr_speed_model.pickle', "rb") as model_file:
        pipeline = pickle.load(model_file)

    # Calculate test score(coefficient of determination).
    test_score = pipeline.score(test_x, test_y)

    print("Test score(coefficient of determination) :", test_score)

    predictVelocity = []


    for i in test_x:
        predictVelocity.append(pipeline.predict([i]))

    x = range(len(test_x))


    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(x,test_y,label="True Velovity",marker='+' )
    ax.plot(x,predictVelocity,label= "Predict Velocity",marker='o' )
    ax.set_title("Prediction")
    ax.set_xlabel("frame")
    ax.set_ylabel("velocity")
    # ax.set_ylim(-1,1)
    ax.legend(loc="best",framealpha=0.5)
    plt.show()



if __name__ == "__main__":
    test()
