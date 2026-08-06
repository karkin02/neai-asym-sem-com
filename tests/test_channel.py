"""Standalone unit tests for the Architecture B channel simulator."""

from __future__ import annotations

import unittest

from architecture_b.channel import CLEAN, EXTREME, PRACTICAL, RESTRICTED, STRESSED, THROTTLED, ChannelConfig, ChannelSimulator, get_channel


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

    def test_throttled_preset_limits_bandwidth_without_drops(self):
        sim = get_channel("throttled", seed=0)
        result = sim.transmit(125_000)
        self.assertEqual(sim.config, THROTTLED)
        self.assertTrue(result.delivered)
        self.assertFalse(result.dropped)
        self.assertAlmostEqual(result.latency_seconds, 25.30)

    def test_restricted_preset_is_bandwidth_bound(self):
        result = get_channel("restricted", seed=0).transmit(5_000)
        self.assertEqual(get_channel("restricted").config, RESTRICTED)
        self.assertTrue(result.delivered)
        self.assertAlmostEqual(result.latency_seconds, 2.30)

    def test_combined_threshold_profiles(self):
        self.assertEqual(get_channel("practical").config, PRACTICAL)
        self.assertEqual(get_channel("stressed").config, STRESSED)
        self.assertEqual(get_channel("extreme").config, EXTREME)
        self.assertEqual(PRACTICAL.bandwidth_bytes_per_sec, 5_000.0)
        self.assertEqual(STRESSED.latency_seconds, 6.0)
        self.assertEqual(EXTREME.drop_probability, 0.50)
        ladder = [get_channel(f"level{i}").config for i in range(1, 6)]
        self.assertEqual([item.bandwidth_bytes_per_sec for item in ladder],
                         [125_000.0, 20_000.0, 5_000.0, 3_000.0, 2_500.0])
        self.assertEqual([item.latency_seconds for item in ladder],
                         [0.50, 1.0, 2.0, 4.0, 6.0])
        self.assertEqual([item.drop_probability for item in ladder],
                         [0.0, 0.05, 0.10, 0.30, 0.50])


if __name__ == "__main__":
    unittest.main()
