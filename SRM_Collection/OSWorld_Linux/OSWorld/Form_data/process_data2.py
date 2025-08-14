import os
import json
import shutil

filepath = "results/pyautogui/screenshot_a11y_tree/gpt-4v"

to_do =  ['00fa164e-2612-4439-992e-157d019a8436', '0326d92d-d218-48a8-9ca1-981cd6d064c7', '0a2e43bf-b26c-4631-a966-af9dfa12c9e5', '0c825995-5b70-4526-b663-113f4c999dd2', '0e5303d4-8820-42f6-b18d-daf7e633de21', '12382c62-0cd1-4bf2-bdc8-1d20bf9b2371', '185f29bd-5da0-40a6-b69c-ba7f4e0324ef', '1954cced-e748-45c4-9c26-9855b97fbc5e', '1d17d234-e39d-4ed7-b46f-4417922a4e7c', '1e8df695-bd1b-45b3-b557-e7d599cf7597', '1f18aa87-af6f-41ef-9853-cdb8f32ebdea', '21ab7b40-77c2-4ae6-8321-e00d3a086c73', '22a4636f-8179-4357-8e87-d1743ece1f81', '26150609-0da3-4a7d-8868-0faf9c5f01bb', '26a8440e-c166-4c50-aef4-bfb77314b46b', '2a729ded-3296-423d-aec4-7dd55ed5fbb3', '2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e', '30e3e107-1cfb-46ee-a755-2cd080d7ba6a', '3a7c8185-25c1-4941-bd7b-96e823c9f21f', '3a93cae4-ad3e-403e-8c12-65303b271818', '3b27600c-3668-4abd-8f84-7bcdebbccbdb', '3d1682a7-0fb0-49ae-a4dc-a73afd2d06d5', '3f05f3b9-29ba-4b6b-95aa-2204697ffc06', '415ef462-bed3-493a-ac36-ca8c6d23bf1b', '42d25c08-fb87-4927-8b65-93631280a26f', '42e0a640-4f19-4b28-973d-729602b5a4a7', '46407397-a7d5-4c6b-92c6-dbe038b1457b', '47f7c0ce-a5fb-4100-a5e6-65cd0e7429e5', '48c46dc7-fe04-4505-ade7-723cba1aa6f6', '4de54231-e4b5-49e3-b2ba-61a0bec721c0', '4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc', '535364ea-05bd-46ea-9937-9f55c68507e8', '5ac2891a-eacd-4954-b339-98abba077adb', '5bc63fb9-276a-4439-a7c1-9dc76401737f', '67890eb6-6ce5-4c00-9e3d-fb4972699b06', '6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a', '6f81754e-285d-4ce0-b59e-af7edb02d108', '72b810ef-4156-4d09-8f08-a0cf57e7cefe', '788b3701-3ec9-4b67-b679-418bfa726c22', '78aed49a-a710-4321-a793-b611a7c5b56b', '7e287123-70ca-47b9-8521-47db09b69b14', '7f35355e-02a6-45b5-b140-f0be698bcf85', '82e3c869-49f6-4305-a7ce-f3e64a0618e7', '881deb30-9549-4583-a841-8270c65f2a17', '897e3b53-5d4d-444b-85cb-2cdc8a97d903', '8e116af7-7db7-4e35-a68b-b0939c066c78', '8ea73f6f-9689-42ad-8c60-195bbf06a7ba', '91190194-f406-4cd6-b3f9-c43fac942b22', '98e8e339-5f91-4ed2-b2b2-12647cb134f4', '9d425400-e9b2-4424-9a4b-d4c7abac4140', '9f3bb592-209d-43bc-bb47-d77d9df56504', 'a0b9dc9c-fc07-4a88-8c5d-5e3ecad91bcb', 'aceb0368-56b8-4073-b70e-3dc9aee184e0', 'b21acd93-60fd-4127-8a43-2f5178f4a830', 'b4f95342-463e-4179-8c3f-193cd7241fb2', 'b52b40a5-ad70-4c53-b5b0-5650a8387052', 'd1acdb87-bb67-4f30-84aa-990e56a09c92', 'd53ff5ee-3b1a-431e-b2be-30ed2673079b', 'd68204bf-11c1-4b13-b48b-d303c73d4bf6', 'da922383-bfa4-4cd3-bbad-6bebab3d7742', 'dd60633f-2c72-42ba-8547-6f2c8cb0fdb0', 'deec51c9-3b1e-4b9e-993c-4776f20e8bb2', 'df67aebb-fb3a-44fd-b75b-51b6012df509', 'e246f6d8-78d7-44ac-b668-fcf47946cb50', 'e8172110-ec08-421b-a6f5-842e6451911f', 'f4aec372-4fb0-4df5-a52b-79e0e2a5d6ce', 'f5c13cdd-205c-4719-a562-348ae5cd1d91', 'f8cfa149-d1c1-4215-8dac-4a0932bad3c2']
# num to do = 68

num = 0

for filedir in os.listdir(filepath):
    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    for filedir2 in os.listdir(os.path.join(filepath, filedir)):
        if filedir2 in to_do:
            num += 1
            shutil.rmtree(os.path.join(filepath, filedir, filedir2))

print("num = ", num)