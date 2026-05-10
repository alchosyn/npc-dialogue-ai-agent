import datetime


def get_current_time() -> str:
    utc_time = datetime.datetime.now(datetime.timezone.utc)
    sg_time = utc_time + datetime.timedelta(hours=8)
    return sg_time.strftime("%Y-%m-%d %H:%M:%S")
