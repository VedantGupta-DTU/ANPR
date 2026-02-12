from edge_sim_py import *
import pandas as pd
import os

# Define a custom collection method for metrics
def custom_collect_method(self) -> dict:
    metrics = {
        "Instance ID": self.id,
        "Power Consumption": self.get_power_consumption(),
        "CPU Usage": self.cpu_demand,
        "RAM Usage": self.memory_demand,
    }
    return metrics

def my_algorithm(parameters):
    # Simplest algorithm: Place all unprovisioned services on the first available server
    for service in Service.all():
        if not service.server:
            # Try to place on the Pi (Server 1)
            for server in EdgeServer.all():
                if server.has_capacity_to_host(service=service):
                    print(f"[Step {parameters['current_step']}] Provisioning {service} on {server}")
                    service.provision(target_server=server)
                    break

def stopping_criterion(model):
    return model.schedule.steps == 10

# Initialize Simulator
simulator = Simulator(
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)

# Assign custom collection
EdgeServer.collect = custom_collect_method

print("Loading dataset 'ocr_sim.json'...")
try:
    simulator.initialize(input_file="ocr_sim.json")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print("Starting simulation (10 steps)...")
simulator.run_model()

# Analyze Results
print("\nSimulation Complete. Results:")
metrics = simulator.agent_metrics["EdgeServer"]
df = pd.DataFrame(metrics)

# Filter for Server 1 (Pi)
pi_metrics = df[df["Instance ID"] == 1]
print(pi_metrics[["Time Step", "Power Consumption", "CPU Usage", "RAM Usage"]])

print("\nSummary:")
avg_power = pi_metrics["Power Consumption"].mean()
print(f"Average Power Consumption: {avg_power:.2f} W")
print(f"Max Power Consumption: {pi_metrics['Power Consumption'].max():.2f} W")
print(f"Detected CPU Demand: {pi_metrics['CPU Usage'].max()} Cores")
