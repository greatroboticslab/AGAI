"""Print the node IPs of a running Ray cluster in the order vLLM 0.32 will give them pipeline-parallel ranks:
the driver's node first, then the other GPU nodes sorted by Ray node ID (vllm/v1/executor/ray_utils.py,
get_bundles_sorted_by_node; one GPU per node here, so one bundle per node). Node IDs are random per `ray start`,
so this has to run after the cluster is up and before the weights are staged.
  python cluster/infra/ray_pp_order.py DRIVER_IP   (with RAY_ADDRESS set)"""
import sys

import ray

ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) >= 1]
driver = [n for n in nodes if n["NodeManagerAddress"] == sys.argv[1]]
assert len(driver) == 1, f"driver {sys.argv[1]} not among GPU nodes {[n['NodeManagerAddress'] for n in nodes]}"
others = sorted((n for n in nodes if n is not driver[0]), key=lambda n: n["NodeID"])
print(" ".join(n["NodeManagerAddress"] for n in driver + others))
