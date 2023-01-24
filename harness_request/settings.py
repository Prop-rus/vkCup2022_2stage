import os

HARNESS_HOST = ''  # insert your service URL
ENGINE_ID = os.getenv('ENGINE_ID', '')  # insert your pretrained engine_id

num_recs = 250

json_to_post = '''{
              "user": "%s",
              "rules": [
              ],
            "num": 250
            }'''
