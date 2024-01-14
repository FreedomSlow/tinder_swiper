import argparse
import datetime
from dateutil.relativedelta import relativedelta
import logging
import os
import pickle
from random import random
from time import sleep

import requests
from tqdm import tqdm

from helpers import read_pickle

KM_IN_MILES = 1.60934


class tinderAPI:
    # TODO: Write proper response handling
    def __init__(self, token: str, api_url: str = "https://api.gotinder.com"):
        self._token = token
        self.api_url = api_url

    def create_profile(self):
        data = requests.get(
            f"{self.api_url}/v2/profile?include=account%2Cuser",
            headers={"X-Auth-Token": self._token}
        ).json()
        return UserProfile(data["data"], self)

    def get_matches(self, limit: int = 10):
        data = requests.get(
            f"{self.api_url}/v2/matches?count={limit}",
            headers={"X-Auth-Token": self._token}
        ).json()
        return list(map(lambda match: UserProfile(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(
            f"{self.api_url}/like/{user_id}",
            headers={"X-Auth-Token": self._token}
        ).json()
        return {
            "match": data["match"],
            "likes_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id: str):
        try:
            requests.get(
                f"{self.api_url}/pass/{user_id}",
                headers={"X-Auth-Token": self._token}
            ).json()
        except Exception as err:
            print(err)
            return False
        return True

    def get_profiles(self):
        data = requests.get(f"{self.api_url}/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda user: UserProfile(user["user"], self), data["data"]["results"]))


class UserProfile:
    def __init__(self, data, api: tinderAPI):
        self.api = api
        self._parse_data(data)

    def _parse_data(self, data):
        self.id = data["_id"]
        # Convert miles to units that normal people use
        self.distance = data.get("distance_mi", 0) / KM_IN_MILES

        self.name = data.get("name", "Unknown")
        self.photos = list(map(lambda photo: photo["url"], data.get("photos", [])))

        self.bio = data.get("bio", "")

        if data.get("birth_date", False):
            self.birth_date = datetime.datetime.strptime(data["birth_date"], "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            self.birth_date = None

        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]
        self.jobs = list(
            map(
                lambda job: {
                    "title": job.get("title", {}).get("name"),
                    "company": job.get("company", {}).get("name")},
                data.get("jobs", [])
            )
        )
        self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

    def bio_to_dict(self):
        return {
            "name": self.name,
            "bio": self.bio,
            "age": int(relativedelta(datetime.datetime.today(), self.birth_date).years)
        }

    def __repr__(self):
        return f"{self.id}: {self.name} ({self.birth_date.strftime('%d.%m.%Y')})"

    def like(self):
        return self.api.like(self.id)

    def dislike(self):
        return self.api.dislike(self.id)

    def download_photos(self, image_dir: str, sleep_for: float = 0.5, already_downloaded: set = None):
        if already_downloaded is not None and self.id in already_downloaded:
            return already_downloaded
        if already_downloaded is None:
            already_downloaded = {self.id}
        else:
            already_downloaded.add(self.id)

        for ind, image_url in enumerate(self.photos):
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                with open(f"{image_dir}/{self.id}_{self.name}_{ind}.jpeg", "wb") as f:
                    f.write(req.content)

            # Necessary to not get banned
            sleep(random() * sleep_for)

        return already_downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="profiles.pkl"
    )
    parser.add_argument(
        "--bios",
        type=str,
        default="bios.pkl"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(20)  # 20 = INFO
    logging.basicConfig(format="%(asctime)s : %(levelname)s | %(message)s")

    image_dir = f"{args.dir}/images"
    os.makedirs(image_dir, exist_ok=True)
    profiles_path = f"{args.dir}/{args.profiles}"
    bios_path = f"{args.dir}/{args.bios}"

    logger.info("Create API connection")
    api = tinderAPI(args.token)

    logger.info("Start parsing profiles")

    already_downloaded = read_pickle(profiles_path)
    bios = read_pickle(bios_path, {})
    # TODO: Make async
    for _ in range(args.num_requests):
        profiles = api.get_profiles()
        for profile in tqdm(profiles):
            already_downloaded = profile.download_photos(image_dir, already_downloaded=already_downloaded)
            sleep(random() * 5)
            bios[profile.id] = profile.bio_to_dict()

    logger.info("Saving parsed profiles and bios")
    with open(bios_path, "wb") as f:
        pickle.dump(bios, f)
    with open(profiles_path, "wb") as f:
        pickle.dump(already_downloaded, f)

    logger.info("All done!")
