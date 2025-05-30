import os
import json
import shutil

filepath = "results/pyautogui/screenshot_a11y_tree/gpt-4v/Windows"
to_do =  ["26660ad1-6ebb-4f59-8cba-a8432dfe8d38", "3a93cae4-ad3e-403e-8c12-65303b271818", "3b27600c-3668-4abd-8f84-7bcdebbccbdb", "455d3c66-7dc6-4537-a39a-36d3e9119df7", "46407397-a7d5-4c6b-92c6-dbe038b1457b", "550ce7e7-747b-495f-b122-acdc4d0b8e54", "5d901039-a89c-4bfb-967b-bf66f4df075e", "6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a", "897e3b53-5d4d-444b-85cb-2cdc8a97d903", "c867c42d-a52d-4a24-8ae3-f75d256b5618"]
# num to do = 10

num = 0

for filedir in os.listdir(filepath):
    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    for filedir2 in os.listdir(os.path.join(filepath, filedir)):

        if filedir2 in to_do:
            num += 1
            shutil.rmtree(os.path.join(filepath, filedir, filedir2))

print("num = ", num)

done =  ["0810415c-bde4-4443-9047-d5f70165a697", "09a37c51-e625-49f4-a514-20a773797a8a", "0b17a146-2934-46c7-8727-73ff6b6483e8", "0e763496-b6bb-4508-a427-fad0b6c3e195", "185f29bd-5da0-40a6-b69c-ba7f4e0324ef", "1f18aa87-af6f-41ef-9853-cdb8f32ebdea", "26150609-0da3-4a7d-8868-0faf9c5f01bb", "2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e", "3aaa4e37-dc91-482e-99af-132a612d40f3", "3ef2b351-8a84-4ff2-8724-d86eae9b842e", "4188d3a4-077d-46b7-9c86-23e1a036f6c1", "4bcb1253-a636-4df4-8cb0-a35c04dfef31", "51b11269-2ca8-4b2a-9163-f21758420e78", "6054afcb-5bab-4702-90a0-b259b5d3217c", "6d72aad6-187a-4392-a4c4-ed87269c51cf", "6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a", "6f81754e-285d-4ce0-b59e-af7edb02d108", "74d5859f-ed66-4d3e-aa0e-93d7a592ce41", "7a4e4bc8-922c-4c84-865c-25ba34136be1", "7efeb4b1-3d19-4762-b163-63328d66303b", "8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14", "8e116af7-7db7-4e35-a68b-b0939c066c78", "9ec204e4-f0a3-42f8-8458-b772a6797cab", "a097acff-6266-4291-9fbd-137af7ecd439", "a82b78bb-7fde-4cb3-94a4-035baf10bcf0", "a9f325aa-8c05-4e4f-8341-9e4358565f4f", "abed40dc-063f-4598-8ba5-9fe749c0615d", "b21acd93-60fd-4127-8a43-2f5178f4a830", "b5062e3e-641c-4e3a-907b-ac864d2e7652", "ce88f674-ab7a-43da-9201-468d38539e4a", "d1acdb87-bb67-4f30-84aa-990e56a09c92", "da52d699-e8d2-4dc5-9191-a2199e0b6a9b", "deec51c9-3b1e-4b9e-993c-4776f20e8bb2", "e2392362-125e-4f76-a2ee-524b183a3412", "e528b65e-1107-4b8c-8988-490e4fece599", "eb03d19a-b88d-4de4-8a64-ca0ac66f426b", "eb303e01-261e-4972-8c07-c9b4e7a4922a", "ecb0df7a-4e8d-4a03-b162-053391d3afaf", "ecc2413d-8a48-416e-a3a2-d30106ca36cb", "f918266a-b3e0-4914-865d-4faa564f1aef"]