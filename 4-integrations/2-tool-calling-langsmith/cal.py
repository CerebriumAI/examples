from datetime import datetime, timedelta

import pytz


def parse_time(time_str):
    """Parse a time string in ISO format to a datetime object in EST timezone."""
    utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    utc_time = utc_time.replace(tzinfo=pytz.utc)  # Set the timezone to UTC
    est_time = utc_time.astimezone(pytz.timezone("America/New_York"))  # Convert to EST
    return est_time


def find_available_slots(api_response, start_date, end_date):
    """Find available time slots within a given date range."""
    busy_times = api_response["busy"]
    working_hours = api_response["workingHours"][
        0
    ]  # Assuming one set of working hours for simplicity

    # Convert start and end times to minutes for easier comparison
    work_start_time = working_hours["startTime"]
    work_end_time = working_hours["endTime"]

    # Initialize the current date to start_date
    current_date = parse_time(start_date)
    end_date = parse_time(end_date)

    available_slots = {}

    while current_date <= end_date:
        if current_date.weekday() + 1 in working_hours["days"]:
            # For each day, start with the assumption that the whole workday is available
            day_slots = [(work_start_time, work_end_time)]

            for busy in busy_times:
                busy_start = parse_time(busy["start"])
                busy_end = parse_time(busy["end"])

                # Check if the busy time is on the current day
                if busy_start.date() == current_date.date():
                    new_day_slots = []
                    for slot_start, slot_end in day_slots:
                        # Convert busy start and end times to minutes
                        busy_start_minutes = busy_start.hour * 60 + busy_start.minute
                        busy_end_minutes = busy_end.hour * 60 + busy_end.minute

                        # Check for overlap and adjust the available slots accordingly
                        if busy_start_minutes > slot_start:
                            new_day_slots.append(
                                (slot_start, min(busy_start_minutes, slot_end))
                            )
                        if busy_end_minutes < slot_end:
                            new_day_slots.append(
                                (max(busy_end_minutes, slot_start), slot_end)
                            )

                    day_slots = new_day_slots

            # Convert slots from minutes back to time strings for readability
            formatted_slots = []
            for slot_start, slot_end in day_slots:
                start_time_str = f"{slot_start // 60:02d}:{slot_start % 60:02d}"
                end_time_str = f"{slot_end // 60:02d}:{slot_end % 60:02d}"
                formatted_slots.append(f"{start_time_str}-{end_time_str}")

            available_slots[current_date.strftime("%Y-%m-%d")] = formatted_slots

        # Move to the next day
        current_date += timedelta(days=1)

    return available_slots
