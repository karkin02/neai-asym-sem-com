import asyncio
import time

# predefined bandwidth conditions
# artificial delay added based on payload size
BANDWIDTH_CONDITIONS = {
    "unlimited": None,
    "1mbps": 1000,
    "100kbps": 100,
    "10kbps": 10,
}

class NetworkThrottle:
    """
    bandwidth_kbps: simulated link speed in kilobits per second (None = unlimited)
    latency_ms:     fixed one-way delay added to every transmission, regardless of size
                    (cellular ping time)             
    """

    def __init__(self, bandwidth_kbps: float = None, latency_ms: float = 0):
        self.bandwidth_kbps = bandwidth_kbps
        self.latency_ms = latency_ms

    async def send(self, payload_bytes: bytes) -> float:
        """
        Simulate sending payload_bytes over the throttled link by 
        literally sleeping for the amount of time it would take
        on a real constrained link
        """
        start = time.time()

        # fixed latency cost
        await asyncio.sleep(self.latency_ms / 1000.0)

        # if bandwidth is limited, add extra delay proportional to payload size
        if self.bandwidth_kbps is not None:
            payload_bits = len(payload_bytes) * 8
            transfer_time_s = payload_bits / (self.bandwidth_kbps * 1000)
            await asyncio.sleep(transfer_time_s)

        elapsed_ms = (time.time() - start) * 1000
        return elapsed_ms

