# using the requests library to access internet data

#import the requests library
# import requests
import json
# from datetime import datetime
import os
import time
import pandas as pd



def main():
    # initialize
    # url = "http://192.168.50.78/dump1090-fa/data/aircraft.json"

# setup timer for < one day
    # now = datetime.now()
    # today = now.date()
    # loop = today

    dataFile = open("flights4.json")
    ads = json.load(dataFile)
    print ads

    # while loop == today:
    #     result = requests.get(url)
    #     dataobj = result.json()
    #     # heartbeat only
    #     print("{0} targets".format(len(dataobj['aircraft'])))

        # 
        # dataFile = open("pi_ADSB.json","a")
        # dataFile.write(json.dumps(dataobj['aircraft']))

        # serialized = open("pi_ADSB_formatted.json","a")
        # serialized.write(json.dumps(dataobj['aircraft'], indent=4))

        # navmodes = open("ADSB_navmodes.json","a")
        # navmodes.write(json.dumps(dataobj['aircraft'.'nav_modes']))

        # now = datetime.now()
        # today = now.date()
        # stat = os.stat("pi_ADSB.json")
        # print(stat.st_size)
        # time.sleep(30)






#     time.sleep(5)
   
    # Use requests to issue a standard HTTP GET request
    # result = requests.get(url)

    # # Use the built-in JSON function to return parsed data
    # dataobj = result.json()
    # message_count = json.dumps(dataobj['messages'])
    
    # # write to file
    # dataFile.write(json.dumps(dataobj['aircraft']))


    # # print(len(dataFile))
    # print("tick")





    # # TODO: 

if __name__ == "__main__":
    main()