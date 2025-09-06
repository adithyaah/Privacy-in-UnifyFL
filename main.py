from flwr.simulation import run_simulation
from ClientApp import client
from ServerApp import server
from flwr.common import Metrics, Context

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    run_simulation(
        client_app=client,
        server_app=server,
        num_supernodes=5,  # Or however many you want to simulate
    )