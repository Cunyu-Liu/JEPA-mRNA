import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_hek_fold_queue_round34 as queue


class HekQueueLockTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        queue.RESULTS = Path(self.directory.name)
        self.popen = mock.patch.object(queue.subprocess, "Popen")
        self.process = self.popen.start()
        self.process.return_value.pid = 123

    def tearDown(self):
        self.popen.stop()
        self.directory.cleanup()

    def command(self):
        return self.process.call_args.args[0]

    def test_new_training_uses_shared_high_memory_lock(self):
        queue.launch("el", 3)
        command = self.command()
        self.assertEqual(command[:3], ["flock", queue.GPU_LOCK, queue.PY])
        self.assertIn(queue.SCRIPT, command)
        self.assertEqual(command[command.index("--fold") + 1], "3")

    def test_resume_uses_same_lock_without_changing_protocol(self):
        queue.launch_resume("el", 3, 3500)
        command = self.command()
        self.assertEqual(command[:3], ["flock", queue.GPU_LOCK, queue.PY])
        self.assertIn(queue.RESUME_SCRIPT, command)
        self.assertEqual(command[command.index("--min-free-mib") + 1], "3500")


if __name__ == "__main__":
    unittest.main()
