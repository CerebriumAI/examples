from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading  # Import threading module


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, action_function, stop_event):
        self.action_function = action_function
        self.stop_event = stop_event  # Add a stop event
        self.last_event_time = time.time()

    def on_any_event(self, event):
        if (
            self.stop_event.is_set()
        ):  # Check if stop event is set before handling any event
            return
        self.last_event_time = time.time()
        self.action_function(event)

    def check_timeout(self, timeout=600):
        if time.time() - self.last_event_time > timeout:
            print("Stopping session due to inactivity")
            self.stop_event.set()


def monitor_directory_changes(path=".", action=None, stop_event=None):
    if stop_event is None:
        stop_event = threading.Event()  # Create a stop event if not provided
    event_handler = ChangeHandler(action, stop_event)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while not stop_event.is_set():  # Use the stop event to break the loop
            time.sleep(1)
            event_handler.check_timeout()
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.stop()
        observer.join()
