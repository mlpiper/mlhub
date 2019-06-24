import os
import json
import argparse
import traceback
from datetime import datetime, timedelta
from parallelm.mlops import mlops
from slackclient import SlackClient
from parallelm.mlops.stats.opaque import Opaque

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="slack token")
    parser.add_argument("--channel-name", help="channel to write messages to")
    parser.add_argument("--alert", help="show only alerts if set to True")
    options = parser.parse_args()
    return options

def get_my_agents():
    """
    Return agent list of current ION node
    :return: Agent list used by current ION node
    """

    # Getting the first agent of ion component "0"
    agent_list = mlops.get_agents(mlops.get_current_node().name)
    if len(agent_list) == 0:
        print("Error - must have agents this ion component is running on")
        raise Exception("Agent list is empty")
    return agent_list

def get_channel_id(sc, name):
    ret = sc.api_call('conversations.list', types='private_channel,public_channel')

    chid = None
    try:
        for ch in ret['channels']:
            if ch['name'] == name:
                print(ch)
                chid = ch['id']
    except Exception:
        print(traceback.format_exc())
    return chid

def main():
    options = parse_args()

    token = options.token
    sc = SlackClient(token)
    is_alert = options.alert == 'True'

    chid = get_channel_id(sc, options.channel_name)
    if chid is None:
        print("Channel <{}> not found".format(options.channel_name))
        return

    mlops.init()

    last_time = -1
    try:
        now = datetime.utcnow()
        hour_ago = (now - timedelta(hours=1))
        node = mlops.get_current_node()

        agents = get_my_agents()

        df = mlops.get_stats(name="last_time_slack",mlapp_node=node.name,
                agent=agents[0].id, start_time=hour_ago,
                end_time=now)
        if df.empty is False:
            last_time = df.iloc[[-1]].value.values[0]
        print("Last message retrieved at {} msec".format(last_time))
    except Exception:
        print(traceback.format_exc())
        pass
    
    if last_time == -1:
        evts = mlops.get_events(is_alert=is_alert)
    else:
        # expects in sec, convert msec to sec
        utc_last = datetime.utcfromtimestamp(last_time/1000)
        evts = mlops.get_events(start_time=utc_last, end_time=datetime.utcnow(), is_alert=is_alert)

    if evts is None or len(evts) == 0:
        print("No events yet found in the time range")
        return

    last_row = evts.iloc[[-1]]
    if last_row.created.values[0] == last_time:
        print("No new events since {} msec".format(datetime.utcfromtimestamp(last_time/1000)))
        return
    else:
        print("New events last {} msec now {} msec".format(
            datetime.utcfromtimestamp(last_time/1000),
            datetime.utcfromtimestamp(last_row.created.values[0]/1000)))


    text = """
[
    {
        "type": "divider"
    },
    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "TEXTKEY"
        },
        "accessory": {
            "type": "image",
            "image_url": "https://media.licdn.com/dms/image/C4E0BAQEHfXCM5IqzeQ/company-logo_400_400/0?e=1559174400&v=beta&t=GxnMnlESAz0XDg9et80kGOrTV6mg0NpxdnttKYs63Jo",
            "alt_text": "ParallelM MCenter"
        }
    }
]"""
    for index,row in evts.iterrows():
        #Sample event
        #{
        #  "clearedTimestamp":0,
        #  "created":1550812504925,
        #  "createdBy":"admin",
        #  "createdTimestamp":1550812504056,
        #  "deletedTimestamp":0,
        #  "description":"KB",
        #  "eventType":"ModelAccepted",
        #  "host":"daenerys-c28",
        #  "id":"9c9ec480-d950-4d4a-acab-774c685c5aa0",
        #  "ionName":"kab slack",
        #  "modelId":"469d47bf-554b-4279-81c9-8dcfb6039494",
        #  "msgType":"UNKNOWN",
        #  "name":"event-2756",
        #  "pipelineInstanceId":"2e0208c1-85c3-41c8-9f68-e2a175d6033d",
        #  "raiseAlert":false,
        #  "reviewedBy":null,
        #  "sequence":2756,
        #  "state":null,
        #  "stateDescription":null,
        #  "type":"ModelAccepted",
        #  "workflowRunId":"1c75a622-05c9-481f-ae87-33a379800b52",
        #  "node":"2"
        #}
        jsons = json.loads(row.to_json())

        if is_alert and jsons['raiseAlert'] is False:
            continue

        if jsons['raiseAlert']:
            alert_message = ":warning:"
        else:
            alert_message = ":ok_hand:"

        text_message = ""

        if jsons['eventType'] == "GenericEvent" and jsons['name'] == "ModelReview":
            text_message = \
                "*MLApp Name*: {}\n" \
                "*Description*: {}\n" \
                "*Event Type*: {}\n" \
                "*Model Id*: {}\n" \
                "*Reviewed By*: {}\n" \
                "*Alert*: {}\n" \
                "*Created At* :clock12: : {}\n" \
                "*Host*: {}\n".format(
                    jsons['ionName'],
                    jsons['stateDescription'],
                    "ModelUpdated",
                    jsons['modelId'],
                    jsons['reviewedBy'],
                    alert_message,
                    datetime.utcfromtimestamp(jsons['created']/1000),
                    jsons['host'])
        else:
            text_message =\
                           "*MLApp Name*: {}\n"\
                           "*Description*: {}\n"\
                           "*Event Type*: {}\n"\
                           "*Message Type*: {}\n"\
                           "*Alert*: {}\n"\
                           "*Created At* :clock12: : {}\n"\
                           "*Host*: {}\n".format(
                                   jsons['ionName'],
                                   jsons['description'],
                                   jsons['eventType'],
                                   jsons['msgType'],
                                   alert_message,
                                   datetime.utcfromtimestamp(jsons['created']/1000),
                                   jsons['host'])

        if jsons['eventType'] == "ModelAccepted":
            text_message = text_message +\
                            "*Model ID*: {}\n".format(jsons['modelId'])

        ret = sc.api_call('chat.postMessage',channel=chid,blocks=text.replace("TEXTKEY", text_message))

    #TODO: Fix this bug - opaques dont work on spark
    #last_time_new = Opaque().name("last_time_slack").data(last_row.created.values[0])
    mlops.set_stat("last_time_slack", last_row.created.values[0].item())
    mlops.done()

if __name__ == "__main__":
    main()
