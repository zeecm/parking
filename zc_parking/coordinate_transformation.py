from typing import Dict, NamedTuple

import numpy as np


class SVY21Coordinates(NamedTuple):
    northing: float
    easting: float


class DegreeCoordinates(NamedTuple):
    latitude: float
    longitude: float


class CoordinateTransformation:
    # source: https://app.sla.gov.sg/sirent/About/PlaneCoordinateSystem
    SEMI_MAJOR_AXIS = 6378137.0
    FLATTENING = 1 / 298.257223563
    ORIGIN_LATITUDE = 1.366666
    ORIGIN_LONGITUDE = 103.833333
    FALSE_NORTHING = 38744.572
    FALSE_EASTING = 28001.642
    SCALE_FACTOR = 1.0

    def __init__(self):
        self.semi_minor_axis = self.SEMI_MAJOR_AXIS * (1 - self.FLATTENING)
        self.eccentricity_squared = self._calculate_eccentricity_squared()
        self.equatorial_arc_consts = self._calculate_equatorial_arc_consts()

    def _calculate_eccentricity_squared(self) -> float:
        return (2 * self.FLATTENING) - (self.FLATTENING * self.FLATTENING)

    def _calculate_equatorial_arc_consts(self) -> Dict[int, float]:
        e_squared = self.eccentricity_squared
        e_fourth = e_squared * e_squared
        e_sixth = e_fourth * e_squared
        coefficient_a1 = 1 - (e_squared / 4) - (3 * e_fourth / 64) - (5 * e_sixth / 256)
        coefficient_a2 = (3.0 / 8.0) * (
            e_squared + (e_fourth / 4) + (15 * e_sixth / 128)
        )
        coefficient_a3 = (15.0 / 256.0) * (e_fourth + (3 * e_sixth / 4))
        coefficient_a4 = 35 * e_sixth / 3072

        coefficient_list = [
            coefficient_a1,
            coefficient_a2,
            coefficient_a3,
            coefficient_a4,
        ]

        return {key + 1: value for key, value in enumerate(coefficient_list)}

    def degrees_to_radians(self, degrees: float) -> float:
        return np.deg2rad(degrees)

    def radians_to_degrees(self, radians: float) -> float:
        return np.rad2deg(radians)

    def convert_lat_lon_to_svy21(
        self, latitude: float, longitude: float
    ) -> SVY21Coordinates:
        latitude_radians = self.degrees_to_radians(latitude)
        longitude_radians = self.degrees_to_radians(longitude)

        sin_latitude = np.sin(latitude_radians)
        cos_latitude = np.cos(latitude_radians)

        meridian_distance = self._calculate_meridian_distance(latitude_radians)
        meridian_distance_origin = self._calculate_meridian_distance(
            self.degrees_to_radians(self.ORIGIN_LATITUDE)
        )

        delta_longitude = longitude_radians - self.degrees_to_radians(
            self.ORIGIN_LONGITUDE
        )

        northing = self._compute_northing(
            sin_latitude,
            cos_latitude,
            meridian_distance,
            meridian_distance_origin,
            delta_longitude,
        )
        easting = self._compute_easting(
            sin_latitude, cos_latitude, meridian_distance, delta_longitude
        )

        return SVY21Coordinates(northing=northing, easting=easting)

    def _compute_northing(
        self,
        sin_latitude: float,
        cos_latitude: float,
        meridian_distance: float,
        meridian_distance_origin: float,
        delta_longitude: float,
    ) -> float:
        coefficient_a2 = self.equatorial_arc_consts[2]

        northing_term1 = (
            delta_longitude**2
            / 2
            * meridian_distance_origin
            * sin_latitude
            * cos_latitude
        )
        northing_term2 = (
            delta_longitude**4
            / 24
            * meridian_distance_origin
            * sin_latitude
            * cos_latitude**3
            * (4 * coefficient_a2**2 + coefficient_a2 - sin_latitude**2)
        )
        northing_term3 = (
            delta_longitude**6
            / 720
            * meridian_distance_origin
            * sin_latitude
            * cos_latitude**5
            * (
                (8 * coefficient_a2**3) * (11 - 24 * sin_latitude * sin_latitude)
                - (28 * coefficient_a2**2) * (1 - 6 * sin_latitude * sin_latitude)
                + coefficient_a2**2 * (1 - 32 * sin_latitude * sin_latitude)
                - coefficient_a2 * 2 * sin_latitude * sin_latitude
                + cos_latitude**4
            )
        )
        northing_term4 = (
            delta_longitude**8
            / 40320
            * meridian_distance_origin
            * sin_latitude
            * cos_latitude**7
            * (
                1385
                - 3111 * sin_latitude * sin_latitude
                + 543 * cos_latitude**4
                - cos_latitude**6
            )
        )

        return self.FALSE_NORTHING + self.SCALE_FACTOR * (
            meridian_distance
            - meridian_distance_origin
            + northing_term1
            + northing_term2
            + northing_term3
            + northing_term4
        )

    def _compute_easting(
        self,
        sin_latitude: float,
        cos_latitude: float,
        meridian_distance: float,
        delta_longitude: float,
    ) -> float:
        coefficient_a2 = self.equatorial_arc_consts[2]

        easting_term1 = (
            delta_longitude**2
            / 6
            * cos_latitude
            * (coefficient_a2 - sin_latitude**2)
        )
        easting_term2 = (
            delta_longitude**4
            / 120
            * cos_latitude**4
            * (
                (4 * coefficient_a2**3) * (1 - 6 * sin_latitude * sin_latitude)
                + coefficient_a2**2 * (1 + 8 * sin_latitude * sin_latitude)
                - coefficient_a2 * 2 * sin_latitude * sin_latitude
                + cos_latitude**4
            )
        )
        easting_term3 = (
            delta_longitude**6
            / 5040
            * cos_latitude**6
            * (
                61
                - 479 * sin_latitude * sin_latitude
                + 179 * cos_latitude**4
                - cos_latitude**6
            )
        )

        return (
            self.FALSE_EASTING
            + self.SCALE_FACTOR
            * meridian_distance
            * delta_longitude
            * cos_latitude
            * (1 + easting_term1 + easting_term2 + easting_term3)
        )

    def _calculate_latitude_from_northing(self, northings: float) -> float:
        coefficient_a2 = self.equatorial_arc_consts[2]
        latitude_radians = self.ORIGIN_LATITUDE * np.pi / 180

        # Iteratively solve for latitude_radians using the northings value
        for _ in range(5):  # Perform 5 iterations for convergence
            sin_latitude = np.sin(latitude_radians)
            radius_of_curvature_prime_vertical = (
                self._calculate_radius_of_curvature_prime_vertical(sin_latitude)
            )

            latitude_term1 = (northings - self.FALSE_NORTHING) / (
                self.SCALE_FACTOR * radius_of_curvature_prime_vertical
            )
            latitude_term2 = (
                latitude_term1
                / (self.SCALE_FACTOR * radius_of_curvature_prime_vertical) ** 3
                * (
                    -coefficient_a2
                    / 6
                    * (
                        1
                        - coefficient_a2**2
                        * (
                            5
                            + 3 * coefficient_a2
                            + 10 * coefficient_a2**2
                            - 4 * coefficient_a2**3
                            - 9 * sin_latitude**2
                        )
                    )
                )
            )  # Coefficients modified for clarity
            latitude_term3 = (
                latitude_term1
                / (self.SCALE_FACTOR * radius_of_curvature_prime_vertical) ** 5
                * (
                    -(coefficient_a2**3)
                    / 120
                    * (
                        5
                        - 18 * coefficient_a2**2
                        + coefficient_a2**4
                        + 14 * sin_latitude**2
                        - 58 * coefficient_a2**2 * sin_latitude**2
                    )
                )
            )
            latitude_term4 = (
                latitude_term1
                / (self.SCALE_FACTOR * radius_of_curvature_prime_vertical) ** 7
                * (
                    -(coefficient_a2**5)
                    / 5040
                    * (
                        61
                        - 479 * coefficient_a2**2
                        + 179 * coefficient_a2**4
                        - coefficient_a2**6
                    )
                )
            )

            latitude_radians = (
                self.ORIGIN_LATITUDE * np.pi / 180
                + latitude_term1
                + latitude_term2
                + latitude_term3
                + latitude_term4
            )

        return latitude_radians

    def _calculate_longitude_from_easting(
        self, eastings: float, latitude_radians: float
    ) -> float:
        coefficient_a2 = self.equatorial_arc_consts[2]
        sec_latitude = 1.0 / np.cos(latitude_radians)
        tangent_latitude = np.tan(latitude_radians)
        tangent_squared_latitude = tangent_latitude * tangent_latitude

        longitude_term1 = eastings / (
            self.SCALE_FACTOR
            * self._calculate_radius_of_curvature(np.sin(latitude_radians))
            * sec_latitude
        )
        longitude_term2 = (
            longitude_term1
            / (
                self.SCALE_FACTOR
                * self._calculate_radius_of_curvature(np.sin(latitude_radians))
                * sec_latitude
            )
            ** 3
            * (coefficient_a2 / 2 * tangent_squared_latitude)
        )
        longitude_term3 = (
            longitude_term1
            / (
                self.SCALE_FACTOR
                * self._calculate_radius_of_curvature(np.sin(latitude_radians))
                * sec_latitude
            )
            ** 5
            * (
                coefficient_a2
                / 24
                * tangent_squared_latitude
                * (
                    5
                    - tangent_squared_latitude
                    + 9 * coefficient_a2
                    + 4 * coefficient_a2**2
                )
            )
        )
        longitude_term4 = (
            longitude_term1
            / (
                self.SCALE_FACTOR
                * self._calculate_radius_of_curvature(np.sin(latitude_radians))
                * sec_latitude
            )
            ** 7
            * (
                coefficient_a2
                / 720
                * tangent_squared_latitude
                * (
                    61
                    + 90 * tangent_squared_latitude
                    + 45 * tangent_squared_latitude**2
                )
            )
        )

        return (
            self.degrees_to_radians(self.ORIGIN_LONGITUDE)
            + longitude_term1
            + longitude_term2
            + longitude_term3
            + longitude_term4
        )

    def convert_svy21_to_lat_lon(
        self, northings: float, eastings: float
    ) -> DegreeCoordinates:
        latitude_radians = self._calculate_latitude_from_northing(northings)
        longitude_radians = self._calculate_longitude_from_easting(
            eastings, latitude_radians
        )

        latitude_degrees = self.radians_to_degrees(latitude_radians)
        longitude_degrees = self.radians_to_degrees(longitude_radians)

        return DegreeCoordinates(latitude=latitude_degrees, longitude=longitude_degrees)

    def _calculate_meridian_distance(self, latitude_radians: float) -> float:
        (
            coefficient_a1,
            coefficient_a2,
            coefficient_a3,
            coefficient_a4,
        ) = self.equatorial_arc_consts.values()
        return self.SEMI_MAJOR_AXIS * (
            coefficient_a1 * latitude_radians
            - coefficient_a2 * np.sin(2 * latitude_radians)
            + coefficient_a3 * np.sin(4 * latitude_radians)
            - coefficient_a4 * np.sin(6 * latitude_radians)
        )

    def _calculate_radius_of_curvature(self, sin_squared_latitude: float) -> float:
        num = self.SEMI_MAJOR_AXIS * (1 - self.eccentricity_squared)
        denom = np.power(
            1 - self.eccentricity_squared * np.sin(sin_squared_latitude), 3.0 / 2.0
        )
        return num / denom

    def _calculate_radius_of_curvature_prime_vertical(
        self, sin_squared_latitude: float
    ) -> float:
        poly = 1 - self.eccentricity_squared * sin_squared_latitude
        return self.SEMI_MAJOR_AXIS / np.sqrt(poly)
