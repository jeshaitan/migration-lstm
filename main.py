import numpy as np

#array of sequences and corresponding final-step-labels
import os
import inspect

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

date = ''
latitude = ''
longitude = ''
temp = ''
masterArray = []
spreadSheetArray = []
row = 0
col = 0
testCount = 0
skipLine = False

myPath = os.path.dirname(os.path.realpath(__file__))
myFile = inspect.getfile(inspect.currentframe())  # script filename (usually with path)
myFile = myFile[myFile.rfind('/') + 1:]
for fn in os.listdir(myPath):
   if os.path.isfile(fn) and fn != myFile and 'csv' in fn:
       date = ''
       lat = ''
       long = ''
       temp = ''
       spreadSheetArray = []
       rowArray = []
       row = 0
       col = 0
       testCount = 0
       skipLine = False
       with open(fn) as animalData:  # Open the file
           animalData.readline()  # Read past the UTC and labels that we don't need
           animalData.readline()  # Read past the UTC and labels that we don't need
           spreadSheetArray = []
           for line in animalData:  # Loop to read every line in file
               if ',,,,' in line:
                   continue
               row += 1
               col = 0  # Restart column count after reading every line
               for word in line.split(','):  # loop through every word in that row
                   if row % 2 == 0:
                       if not skipLine:  # for some reason there's an extra line
                           skipLine = True
                       elif skipLine:
                           skipLine = False
                           continue
                       temp = line.split(',')[1]
                       temp = temp[:temp.find('\\') - 1]
                       if temp != '' and ('-2' not in temp) and temp != '-1' and 'FAL' not in temp:
                           spreadSheetArray.append([float(latitude), float(longitude), float(temp)])
                       continue
                   elif col == 7:
                       longitude = word
                   elif col == 8:
                       latitude = word
                   col += 1
               testCount += 1
               if testCount == -10:
                   break
       masterArray.extend([spreadSheetArray[i:i+10] for i in xrange(0, len(spreadSheetArray), 10)])

seqs = masterArray
steps = map(lambda l: l[len(l) - 1][:2], seqs)

########################################################################################################################

import mechanize
from bs4 import BeautifulSoup
import requests

br = mechanize.Browser()
br.set_handle_redirect(True)
br.addheaders = [('User-agent', 'Firefox')]
br.set_handle_robots(False)  # ignore robots
br.set_handle_refresh(False)  # can sometimes hang without this

def findClose(tempList):
   print '12'
   midIndex = int(len(tempList) / 2)
   firstNum = -1
   secondNum = -1
   val1 = -1
   val2 = -1
   print '13'
   for num in range(midIndex, 0, -1):
       print '14'
       if (tempList[num]):
           print '15'
           firstNum = num
           val1 = tempList[num]
           print '16'
           break
   for num in range(midIndex, len(tempList)):
       print '17'
       if (tempList[num]):
           print '18'
           secondNum = num
           val2 = tempList[num]
           print '19'
           break
   print '20'
   if (val1 == -1 and val2 == -1):
       return False
   print '21'
   if (val1 == -1):
       return val2
   print '22'
   if (val2 == -1):
       return val1
   print '23'
   if ((midIndex - firstNum) < (secondNum - midIndex)):
       return val1
   print '24'
   return val2


def reqtemp(lat, long, time, extraLat):
   extraCount = extraLat * 200
   middleValue = (extraCount / 2) + 1
   endValue = middleValue * 4
   tdCount = 0
   # generate url
   urlTemp = "http://coastwatch.pfeg.noaa.gov/erddap/griddap/"  # generic format
   urlTemp += 'jplMURSST.xhtml?analysed_sst'
   appendUrl = '[(' + date + ')]'  # date
   urlTemp += appendUrl
   botLat = str(lat - extraLat)
   topLat = str(lat + extraLat)
   botLong = str(long - extraLat)
   topLong = str(long + extraLat)
   appendUrl = '[(' + botLat + '):1:(' + topLat + ')][(' + botLong + '):1:(' + topLong + ')]'  # temps
   urlTemp += appendUrl
   try:
       br.open(urlTemp)
   except:
       return -1
   htm = requests.get(br.geturl())
   pageText = BeautifulSoup(htm.text, 'lxml')
   tempList = []
   for table in pageText.findAll("table"):
       numCount = 0
       for sItem in table.findAll("td"):
           tdCount += 1
           numCount += 1
           thisWord = sItem.getText().strip()
           if ('-' in thisWord):
               numCount = 0
           elif (numCount == 3 and 'o' not in thisWord):
               tempList.append(sItem.getText().strip())
               # if tdCount % 4 == 0:
               #     tempList.append(sItem.getText().strip())
   if not any(tempList):  # if the array is empty
       if (extraLat * 2 == 6.4):
           return -2
       return reqtemp(lat, long, time, extraLat * 2)
       # list is empty
   else:
       return findClose(tempList)

########################################################################################################################
max_sequence_length = len(max(seqs,key=len))

#pad and prepare sequences for model
padded_training_seqs = np.array(map(lambda l: l + [[0,0,0]] * (max_sequence_length - len(l)), seqs))
training_final_steps = np.array(steps)
print padded_training_seqs.shape, training_final_steps.shape

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Masking, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

#build and train model
in_dimension = 3
hidden_neurons = 300
out_dimension = 2

model = Sequential()
model.add(BatchNormalization(input_shape=((max_sequence_length, in_dimension))))
model.add(Masking([0,0,0], input_shape=(max_sequence_length, in_dimension)))
model.add(LSTM(hidden_neurons, activation='softmax', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(out_dimension, activation='linear'))

model.compile(loss="mse", optimizer="sgd")
model.fit(padded_training_seqs, training_final_steps, nb_epoch=5, batch_size=1)

#fetch temperature from NOAA and inject into location tuple
def loc_with_temp(ll, i):
    #time_since_initial = average_timestep * i
    #temp = reqtemp(ll[0], ll[1], initial_time + time_since_initial, 0.1)
    temp = reqtemp(ll[0], ll[1], '2012-01-02T02:16:24Z', 0.1)
    return [ll[0], ll[1], temp]

#generate new sequence based on seed
seed_lat = 42.966
seed_long = 39.869
seed_temp = 25.066

current_generated_sequence = np.array([[[seed_lat, seed_long, seed_temp]] + [[0,0,0]] * (max_sequence_length - 1)], dtype=np.dtype(float))

for i in range(0, max_sequence_length - 1):
    next_step = model.predict(current_generated_sequence, batch_size=1, verbose=1)[0]
    current_generated_sequence[0][i + 1] = loc_with_temp(next_step, i)
    #print new iteration
    print(current_generated_sequence[0][i])

np.set_printoptions(threshold=np.nan)
print(current_generated_sequence)
