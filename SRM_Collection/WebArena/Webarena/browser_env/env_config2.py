# websites domain
import os

SHOPPING = os.environ.get("SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")

# print("SHOPPING = ", SHOPPING)
# print("SHOPPING_ADMIN = ", SHOPPING_ADMIN)

assert (
        SHOPPING
), (
    f"Please setup the URLs to each site. Current: \n"
    + f"Shopping: {SHOPPING}\n"
    + f"Shopping Admin: {SHOPPING_ADMIN}\n"
)


# Example account configuration
# To configure accounts, replace the empty dictionary below with actual
# account details in the following format:
#
# ```python
# {
#     "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
#     "gitlab": {"username": "byteblaze", "password": "hello1234"},
#     "shopping": {
#         "username": "emma.lopez@gmail.com",
#         "password": "Password.123",
#     },
#     "shopping_admin": {"username": "admin", "password": "admin1234"},
#     "shopping_site_admin": {"username": "admin", "password": "admin1234"},
# }
# ```
ACCOUNTS = {}

URL_MAPPINGS = {
    SHOPPING: "http://onestopmarket.com",
    SHOPPING_ADMIN: "http://luma.com/admin",
}
