import json

import requests


class ServerPost:
    TYPE_STRESS = "stress"
    TYPE_ACTIVITY = "act"
    TYPE_RESPIRATION = "resp"

    def __init__(self, server, port):
        self.servername = server
        self.serverport = port

    def __getInformation(self, endpoint, token):
        headers = {"token": token}
        r = requests.get(
            self.servername + ":" + str(self.serverport) + "/" + endpoint,
            headers=headers,
        )
        return r.text

    def uploadInformation(self, endpoint, jsondata, token):
        headers = {"token": token, "Content-Type": "application/json"}
        print(
            str(jsondata)
            + " -- "
            + str(self.servername + ":" + str(self.serverport) + "/" + endpoint)
        )
        r = requests.post(
            self.servername + ":" + str(self.serverport) + "/" + endpoint,
            data=json.dumps(jsondata),
            headers=headers,
        )
        return r.text

    def save(self, type, value, babyid, token, name="", comments="", anomaly=False):
        data = {}
        data["name"] = name
        data["comments"] = comments
        data["anomaly"] = anomaly
        data["type"] = type
        data["value"] = '{"value":' + str(value) + "}"
        data["babyid"] = babyid
        print(data)
        self.__uploadInformation("event", data, token)
