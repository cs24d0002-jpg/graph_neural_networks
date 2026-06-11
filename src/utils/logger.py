import sys
from datetime import datetime

class Logger:
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w') if log_file else None
        # Write start time
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write(f"=== Run started at {start_time} ===\n")

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()