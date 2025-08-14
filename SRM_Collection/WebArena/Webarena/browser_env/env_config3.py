# websites domain
import os


# WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
WIKIPEDIA = "http://localhost::8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

assert (
    WIKIPEDIA
), (
    f"Please setup the URLs to each site. Current: \n"
    + f"Wikipedia: {WIKIPEDIA}\n"
)


ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}

URL_MAPPINGS = {
    WIKIPEDIA: "http://wikipedia.org",
}
