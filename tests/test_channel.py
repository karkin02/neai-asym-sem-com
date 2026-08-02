"""Standalone unit tests for the Architecture B channel simulator."""

from __future__ import annotations

import unittest

from architecture_b.channel import CLEAN, ChannelConfig, ChannelSimulator, get_channel


class ChannelTest(unittest.TestCase):
    def test_clean_channel_delivers_and_bills_bytes(self):
        sim = ChannelSimulator(CLEAN, seed=0)
        result = sim.transmit(10_000)
        self.assertTrue(result.delivered)
        self.assertFalse(result.dropped)
        self.assertEqual(result.payload_bytes, 10_000)
        self.assertEqual(result.condition, "clean")

    def test_bandwidth_adds_transfer_time(self):
        cfg = ChannelConfig("slow", bandwidth_bytes_per_sec=1000.0, latency_seconds=0.1)
        sim = ChannelSimulator(cfg, seed=0)
        result = sim.transmit(2000)  # 2000 / 1000 = 2.0s transfer + 0.1 latency
        self.assertAlmostEqual(result.latency_seconds, 2.1, places=6)

    def test_forced_drop_and_forced_delivery(self):
        drop = ChannelSimulator(ChannelConfig("d", 125_000, 0.3, drop_probability=1.0))
        self.assertFalse(drop.transmit(500).delivered)
        self.assertTrue(drop.transmit(500).dropped)
        keep = ChannelSimulator(ChannelConfig("k", 125_000, 0.3, drop_probability=0.0))
        self.assertTrue(keep.transmit(500).delivered)

    def test_get_channel_rejects_unknown(self):
        with self.assertRaises(ValueError):
            get_channel("teleport")


if __name__ == "__main__":
    unittest.main()
