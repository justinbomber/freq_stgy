import os


# traverse func that can take of "-futures" name behind every json file
# ex: from 'BTC_USDT-1h-futures.json' this name to 'BTC_USDT-1h.json' this name

def traverse(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                name = file.split("-futures")[0]
                os.rename(os.path.join(root, file), os.path.join(root, name))

# the dir name is 'C:\Users\justin\Desktop\data\jsonile\binance'


# a func that can change every file under C:\Users\justin\Desktop\data\jsonile\binance into json file
def change_to_json(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(root, file.split(".")[0] + ".json"))
change_to_json('C:\\Users\\justin\\Desktop\\data\\jsonile\\binance')

