import os
import csv
import pandas as pd
import json
import chardet

def ask_chatgpt_ant(messages):
    param = get_default_config(model="gpt-4o")
    param["queryConditions"]["model"] = "gpt-4o"
    param["queryConditions"]["temperature"] = "0.2"

    param["queryConditions"]["messages"] = messages
    try:
        response = ask_chatgpt(param)
        # print(response)
        return response
    except Exception as e:
        print("error: ", e)
        return False


system_prompt = '''
    Compare the following two strings and determine if they are either identical or highly similar. Consider the following aspects:
    
    1. **Text Normalization**:
       - Ignore case differences (e.g., "Apple" vs "apple")
       - Ignore minor punctuation variations (e.g., "don't" vs "dont")
       - Ignore extra whitespace (e.g., "hello world" vs "hello  world")
    
    2. **Semantic Similarity**:
       - Consider synonyms and abbreviations (e.g., "doctor" vs "dr")
       - Account for common misspellings (e.g., "accomodate" vs "accommodate")
       - Recognize number/letter substitutions (e.g., "5" vs "five")
    
    3. **Content Matching**:
       - Focus on core meaning rather than exact wording
       - Consider word order variations (e.g., "cat and dog" vs "dog and cat")
       - Ignore non-essential words (articles, prepositions)
    
    Format requirements:
    - Return JSON format with:
      a) Boolean 'identical' flag
      b) Boolean 'highly_similar' flag
      c) Similarity score (0-100)
      d) Brief explanation
    
    Examples of good responses:
    {
      "identical": true,
      "highly_similar": true,
      "similarity_score": 100,
      "explanation": "Exact match after normalization"
    }
    
    {
      "identical": false,
      "highly_similar": true,
      "similarity_score": 92,
      "explanation": "Minor punctuation differences with identical core content"
    }
    
    The formation of two strings is as follows:
    String 1: [INSERT_FIRST_STRING]
    String 2: [INSERT_SECOND_STRING]
'''

user_prompt = """
String 1: <INSERT_FIRST_STRING>
String 2: <INSERT_SECOND_STRING>
"""

def compare_similarity(text1, text2):
    user_message = user_prompt
    user_message = user_message.replace("<INSERT_FIRST_STRING>", text1)
    user_message = user_message.replace("<INSERT_SECOND_STRING>", text2)

    # print("type:", row['type'])
    # print("\n")
    #
    # if (row['type'] != 'traj'):
    #     print("chosen:\n", row['chosen'])
    #     # print("\n")
    #     print("rejected:\n", row['rejected'])
    #     print("\n")

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                },

            ]
        }
    ]

    # print("messages:\n", messages)
    # print("\n")

    while True:
        response = ask_chatgpt_ant(messages)
        if (response != False):
            print(response)
            print("\n")
            if response[0] == '`':
                response = response[8:-4]
            response = json.loads(response)
            break

    return response



list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"] # , "SRM_5.csv", "SRM_6.csv"
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']


# final_csv_file = "SRM_test_ok_modify.csv"
# final_csv_file = "SRM_test_ok_modify_replace.csv"
final_csv_file = "SRM_test_ok_modify3.csv"
# if not os.path.exists(final_csv_file):
#     with open(final_csv_file, 'w', newline='') as file:
#         fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
#         writer = csv.DictWriter(file, fieldnames=fields)
#         writer.writeheader()

df_final = pd.read_csv(final_csv_file, encoding="GB2312")

# df_final = df_final.loc[df_final['result'] == 1]

df_final.drop_duplicates(subset=['instruction', 'step_idx', 'type', 'reason_steps', 'chosen', 'rejected'], keep='first', inplace=True)

# df_final.to_csv("SRM_test_ok_modify_filter.csv", index=False)
# df_final.to_csv("SRM_test_ok_modify_replace_filter.csv", index=False, encoding="GB2312")
df_final.to_csv("SRM_test_ok_modify3_filter.csv", index=False, encoding="GB2312")
