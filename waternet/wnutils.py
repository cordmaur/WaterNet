from pathlib import Path


def parse_maja_name(name: str):
    """
    Get the string with a MAJA img name (THEIA format) and extract the useful information from it.
    @param name: Image name
    @return: Dictionary with the values.
    """

    # ignore extension
    name = name.split(".")[0]

    mission, datetime, level, tile, metadata, version = name.split("_")
    date, time, _ = datetime.split("-")

    result = dict(
        mission="S" + mission[-2:],
        level=level,
        datetime=f"{date}T{time}",
        tile=tile[1:],
        date=date,
        time=time,
        year=date[:4],
        month=date[4:6],
        day=date[6:],
    )
    return result


def parse_s2cor_name(name):
    sat, sensor, full_datetime, _, orbit, tile, _ = name.split("_")
    date, time = full_datetime.split("T")
    result = {
        "satellite": sat,
        "tile": tile[1:],
        "orbit": orbit,
        "sensor": sensor,
        "date": date,
        "year": date[:4],
        "month": date[4:6],
        "day": date[6:],
        "time": time,
        "iso_datetime": date + "T" + time,
    }

    return result


def parse_planetary_name(name):
    sat, sensor, full_datetime, _, tile, _ = name.split("_")
    date, time = full_datetime.split("T")
    result = {
        "satellite": sat,
        "tile": tile[1:],
        "sensor": sensor,
        "date": date,
        "year": date[:4],
        "month": date[4:6],
        "day": date[6:],
        "time": time,
        "iso_datetime": date + "T" + time,
    }

    return result


def parse_sat_name(folder, img_type="MAJA"):
    img = Path(folder)
    try:
        if img_type.upper() in ["MAJA", "THEIA"]:
            return parse_maja_name(img.stem)
        elif img_type.upper() == "S2COR":
            return parse_s2cor_name(img.stem)
        elif img_type.upper() == "PLANETARY":
            return parse_planetary_name(img.stem)
    except Exception as e:
        print(e)
        return None
