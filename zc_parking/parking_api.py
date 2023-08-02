import json
import traceback
from abc import ABC
from typing import Any, Dict, Optional

import pandas as pd
import requests
from loguru import logger

from zc_parking.constants import LOT_TYPE_MAPPING


class ParkingAPI(ABC):
    _headers: Dict[str, str]

    def _make_api_call(
        self,
        endpoint: str,
        additional_headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str | int]] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        params = params or {}
        additional_headers = additional_headers or {}

        logger.info(
            f"making 1 api call to {endpoint} with params {params}, timeout {timeout}"
        )
        auth_header = self._headers | additional_headers
        response = requests.get(
            endpoint, headers=auth_header, params=params, timeout=timeout
        )

        if response.status_code != 200:
            logger.error(f"failed to retrieve data from {endpoint}")
            return {"error": response.text}

        return self._text_json_to_dict(response.text)

    def _text_json_to_dict(self, text: str) -> Dict[str, Any]:
        try:
            return dict(json.loads(text))
        except json.JSONDecodeError:
            logger.error(f"failed to decode json: {traceback.format_exc()}")
            return {"failed to decode": text}


class URAParkingAPI(ParkingAPI):
    TOKEN_ENDPOINT = "https://www.ura.gov.sg/uraDataService/insertNewToken.action"
    CARPARK_LOTS_ENDPOINT = "https://www.ura.gov.sg/uraDataService/invokeUraDS?service=Car_Park_Availability"
    CARPARK_LIST_RATES_ENDPOINT = (
        "https://www.ura.gov.sg/uraDataService/invokeUraDS?service=Car_Park_Details"
    )

    def __init__(self, access_key: str, token: Optional[str] = None):
        self._headers = {
            "AccessKey": access_key,
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
            "x-requested-with": "XMLHttpRequest",
        }
        token = token or self._get_token()
        self._headers |= {"Token": token}

    def _make_api_call(
        self,
        endpoint: str,
        additional_headers: Dict[str, str] | None = None,
        params: Dict[str, str | int] | None = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        result = super()._make_api_call(endpoint, additional_headers, params, timeout)
        if result["Status"] != "Success":
            logger.warning(f"failed to retrieve data from {endpoint}")
        return result

    def _get_token(self, timeout: int = 5) -> str:
        logger.info("attempting to authenticate with URA api...")

        response = requests.get(
            self.TOKEN_ENDPOINT, headers=self._headers, timeout=timeout
        )

        if response.status_code != 200:
            logger.error(response.text)
            raise ValueError("Failed to Authenticate with URA API")

        result = self._text_json_to_dict(response.text)

        if result["Status"] != "Success":
            logger.warning(f"Failed to retrieve token, error message: {result}")
        else:
            logger.info("authentication success")

        return result.get("Result", "")

    def get_carpark_lots(self) -> pd.DataFrame:
        carpark_lots_response = self._make_api_call(endpoint=self.CARPARK_LOTS_ENDPOINT)
        if carpark_lots := carpark_lots_response.get("Result"):
            df = pd.DataFrame(carpark_lots)
            return self._process_lots_data(df)
        return pd.DataFrame()

    def _process_lots_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["lotType"] = data["lotType"].map(LOT_TYPE_MAPPING)
        return data

    def _process_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.explode("geometries", ignore_index=True)
        data["geometries"] = data["geometries"].apply(
            lambda x: dict(x).get("coordinates", "0,0")
        )
        # TODO: convert SVY21 to LatLon
        return data

    def get_carpark_list_and_rates(self) -> pd.DataFrame:
        carpark_list_and_rates_response = self._make_api_call(
            endpoint=self.CARPARK_LIST_RATES_ENDPOINT
        )
        if list_and_rates := carpark_list_and_rates_response.get("Result"):
            return pd.DataFrame(list_and_rates)
        return pd.DataFrame()


class LTAParkingAPI(ParkingAPI):
    PARKING_AVAILABILITY_ENDPOINT = (
        "http://datamall2.mytransport.sg/ltaodataservice/CarParkAvailabilityv2"
    )

    def __init__(self, account_key: str):
        self._headers = {"AccountKey": account_key}

    def get_parking_availability(self) -> pd.DataFrame:
        data = self._make_api_call(endpoint=self.PARKING_AVAILABILITY_ENDPOINT)
        if "value" in data:
            df = pd.DataFrame(data["value"])
            return self._process_parking_availability_data(df)
        logger.warning(f"unexpected data: {data}")
        return pd.DataFrame()

    def _process_parking_availability_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["LotType"] = data["LotType"].map(LOT_TYPE_MAPPING)
        return self._filter_for_only_lta(data)

    def _filter_for_only_lta(self, data: pd.DataFrame):
        only_lta = data["Agency"] == "LTA"
        return data[only_lta]
