import unittest

from run_linearpartition_19000_cpu import MINIMUM_AVAILABLE_BYTES, memavailable_bytes


class HoldoutRunTests(unittest.TestCase):
    def test_memavailable_is_parsed_in_bytes(self):
        self.assertEqual(memavailable_bytes("MemTotal: 100 kB\nMemAvailable: 8388608 kB\n"),
                         MINIMUM_AVAILABLE_BYTES)
        with self.assertRaises(ValueError):
            memavailable_bytes("MemFree: 100 kB\n")


if __name__ == "__main__":
    unittest.main()
