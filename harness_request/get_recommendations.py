import json

import asyncio
import aiohttp

from datetime import datetime
import random

import settings


res = []


async def get_recs_async(subscribers):
    "creates iohttp session, creaes the list of post requests, appends the answers in result list"

    print('start')
    cur_time = datetime.now()
    host = settings.HARNESS_HOST
    engine_id = settings.ENGINE_ID
    url = f'{host}/engines/{engine_id}/queries'
    timeout = aiohttp.ClientTimeout(total=7200, connect=3600, sock_connect=3600, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        post_tasks = []
        # prepare the coroutines that post
        for user in subscribers:
            post_tasks.append(do_post(session, url, user))
        # now execute them all at once
        await asyncio.gather(*post_tasks)

    print('finish', datetime.now() - cur_time)
    print('number of recs = ', len(res))

    return res


async def do_post(session, url, user):
    "posts one request by user"

    async with session.post(url,
                            json=json.loads(settings.json_to_post % (user))) as response:
        data = await response.text()
        try:
            response_list = json.loads(data)['result']
            # print("-> Got rec for ", user)
            recomend_list = []
            score_list = []
            for rec_item in response_list:
                recomend_list.append(int(rec_item['item']))
                score_list.append(rec_item['score'])
            if settings.shuffle:
                random.shuffle(recomend_list)
            res.append(str(user) +
                       ';' +
                       ','.join(list(map(str, recomend_list[:settings.num_recs]))) +
                       ';' +
                       ','.join(list(map(str, score_list[:settings.num_recs]))))
        except ValueError as e:
            print('failed to load json: ', user, e)


def get_recommendations_from_harness(subscribers):
    "forms the pool of post requests and sends them to harness service in async mode. Returns the list with recommendations for each user"

    res = asyncio.run(get_recs_async(subscribers))

    return res
