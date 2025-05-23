"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json

from browser_env.env_config import *


def main() -> None:
    with open("config_files/test.raw.json", "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    with open("config_files/test.json", "w") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    list = []
    for idx, item in enumerate(data):
        # print("item = ", item)
        # if (item['sites'] == ['shopping']): # 187
        # if (item['sites'] == ['shopping_admin']): # 182
        # if (item['sites'] == ['reddit']): # 106
        if (item['sites'] == ['gitlab']): # 180
        # if (item['sites'] == ['map']):  # 109

        # if ('shopping' in item['sites']):  # 192
        # if ('shopping_admin' in item['sites']):  # 184
        # if ('reddit' in item['sites']):  # 129
        # if ('gitlab' in item['sites']):  # 204
        # if ('map' in item['sites']):  # 128
        # if ('wikipedia' in item['sites']):    h # 23
            # print("222")
            with open(f"config_files/{idx}.json", "w") as f:
                json.dump(item, f, indent=2)
            list.append(idx)
        else:
            # print("111")
            # print("item['sites'] = ", item['sites'])
            continue

    print("total: ", len(list))
    print(list)

if __name__ == "__main__":
    main()
