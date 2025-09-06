from flwr.client import NumPyClient, ClientApp, Client  # ✅ Add this
from flwr.common import Context
from model import Net, get_parameters, set_parameters, train, test
from model import load_datasets
_=load_datasets(partition_id=0)  # Will download & cache CIFAR-10
import torch
from ptflops import get_model_complexity_info
import wandb
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        print("Client CUDA available?", torch.cuda.is_available())
        self.net = net
        self.trainloader = trainloader  
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)
    
    macs, params = get_model_complexity_info(
            net, 
            (3, 32, 32),   # <-- input shape (depends on your dataset, e.g. CIFAR-10 is (3, 32, 32))
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
    print(f"FLOPs: {macs}, Parameters: {params}")

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)