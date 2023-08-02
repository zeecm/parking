import pandas as pd

from zc_parking.parking_api import URAParkingAPI


def test_get_carpark_lots():
    api = URAParkingAPI("")
    lots = api.get_carpark_lots()
    assert isinstance(lots, pd.DataFrame)
