"""Script to automatically login each website"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright

from browser_env.env_config import (
    ACCOUNTS,
    SHOPPING,
    SHOPPING_ADMIN,
)

HEADLESS = True
SLOW_MO = 0


SITES = ["shopping", "shopping_admin"]
URLS = [
    f"{SHOPPING}/wishlist/",
    f"{SHOPPING_ADMIN}/dashboard",
]
EXACT_MATCH = [True, True]
KEYWORDS = ["", "Dashboard"]

# SITES = ["shopping", "shopping_admin"]
# URLS = [
#     f"{SHOPPING}/wishlist/",
#     f"{SHOPPING_ADMIN}/dashboard",
# ]
# EXACT_MATCH = [True, True]
# KEYWORDS = ["", "Dashboard"]


def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    print("d_url = ", d_url)
    # print("content = ", content)
    if keyword:
        print("keyword not in content:", keyword not in content)
        return keyword not in content
    else:
        if url_exact:
            print("url_exact d_url != url:", d_url != url)
            return d_url != url
        else:
            print("url_exact")
            return url not in d_url


def renew_comb(comb: list[str], auth_folder: str = "./.auth") -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    # print("comb = ", comb)

    if "shopping" in comb:
        # print("3333")
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        # print("now1 = ", f"{SHOPPING}/customer/account/login/")
        page.goto(f"{SHOPPING}/customer/account/login/")
        # print("5555")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()


    if "shopping_admin" in comb:
        # print("4444")
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        # print("now2 = ", f"{SHOPPING_ADMIN}")
        # print("SHOPPING_ADMIN = ", SHOPPING_ADMIN)
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    # print("2222")

    print(context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json"))

    context_manager.__exit__()


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def main(auth_folder: str = "./.auth") -> None:
    pairs = list(combinations(SITES, 2))

    # print("pairs = ", pairs)

    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pair in pairs:
            # TODO[shuyanzh] auth don't work on these two sites
            if "reddit" in pair and (
                "shopping" in pair or "shopping_admin" in pair
            ):
                continue
            # print("pair = ", pair)
            executor.submit(
                renew_comb, list(sorted(pair)), auth_folder=auth_folder
            )

        for site in SITES:
            # print("site = ", site)
            executor.submit(renew_comb, [site], auth_folder=auth_folder)

    # print("1111")
    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    # print("cookie_files = ", cookie_files)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for c_file in cookie_files:
            comb = get_site_comb_from_filepath(c_file)
            for cur_site in comb:
                url = URLS[SITES.index(cur_site)]
                keyword = KEYWORDS[SITES.index(cur_site)]
                match = EXACT_MATCH[SITES.index(cur_site)]
                print("Path(c_file) = ", Path(c_file))
                print("url = ", url)
                print("keyword = ", keyword)
                print("match = ", match)
                future = executor.submit(
                    is_expired, Path(c_file), url, keyword, match
                )
                futures.append(future)

    # print("futures = ", futures)

    for i, future in enumerate(futures):
        # print("future.result() = ", future.result())
        assert not future.result(), f"Cookie {cookie_files[i]} expired."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    args = parser.parse_args()
    if not args.site_list:
        main()
    else:
        if "all" in args.site_list:
            main(auth_folder=args.auth_folder)
        else:
            renew_comb(args.site_list, auth_folder=args.auth_folder)