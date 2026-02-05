from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class TrafficMatrix:
    # demand_mbps[(src,dst)] = Mbps
    demand_mbps: Dict[Tuple[int,int], float]
