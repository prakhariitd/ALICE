# -*- coding: utf-8 -*-
"""SourceTrackingDatasetBuild.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TkZgyU6yncOoY81FsOx0SsSEJzyYNi2b
"""

# Build Source Tracking Datasets
import sqlite3
import os

# Clear Datasets
def resetDataset():
    if os.path.exists(dsName):
        os.remove(dsName)
    else:
        print("The file does not exist")

# IMPORTANT GLOBAL VARIABLES
dsName = "source.db"
usersDSname = "users"
tagDSname = "hashtags"

# Create two tables
def createDataset():
    # connecting to the database
    connection = sqlite3.connect(dsName)

    # cursor
    cur = connection.cursor()

    cur.execute('''CREATE TABLE ''' + usersDSname + ''' 
                   (id varchar unique, unclassified, true, false, irrelevant)''')

    cur.execute('''CREATE TABLE ''' + tagDSname + ''' 
                   (id varchar unique, unclassified, true, false, irrelevant)''') # id is the topic, str type
  
    # close the connection
    connection.commit()
    connection.close()

# Check if record (id = did) exists in the table tableName
def isExist(tableName, did):
    # connecting to the database
    connection = sqlite3.connect(dsName)

    # cursor
    cur = connection.cursor()

    cur.execute("SELECT * FROM " + tableName + " WHERE id = '%s'" % did)
    record = cur.fetchone()

    # close the connection
    connection.commit()
    connection.close()
    if record is None:
        return False
    else:
        return True

# Insert a new entry (id = did) in the table tableName
def insertNew(tableName, did):
    prob = [0.0,0.0,0.0,0.0]
    l = [did] + prob

    # connecting to the database
    connection = sqlite3.connect(dsName)

    # cursor
    cur = connection.cursor()

    cur.execute("INSERT INTO " + tableName + " VALUES (?,?,?,?,?)",l)

    # close the connection
    connection.commit()
    connection.close()

# update score
def updateScore(tableName, did, prob):
    

    # connecting to the database
    connection = sqlite3.connect(dsName)

    # cursor
    cur = connection.cursor()
    
    cur.execute("SELECT * FROM " + tableName + " WHERE id = '%s'" % did)
    
    record = cur.fetchone()
    oldScore = record[1:]
    newScore = [sum(value) for value in zip(prob, oldScore)]
    l = newScore + [did]

    cur.execute("UPDATE " + tableName + " SET unclassified=?,true=?,false=?,irrelevant=? WHERE id=?",l)

    # close the connection
    connection.commit()
    connection.close()

# 1. public function
def addFinalResult(tableName, did, prob):
    if not isExist(tableName, did):
        insertNew(tableName, did)
    updateScore(tableName, did, prob)



# Get score
def normalizeScore(score):
    score = list(score)
    s = sum(score)
    for i in range(len(score)):
        score[i] = score[i]/s
    return score
    
# 2. public function
def getScore(tableName, did):

    # connecting to the database
    connection = sqlite3.connect(dsName)

    # cursor
    cur = connection.cursor()

    cur.execute("SELECT * FROM " + tableName + " WHERE id = '%s'" % did)
    record = cur.fetchone()
    sourceTrackingScore = record[1:]

    # close the connection
    connection.commit()
    connection.close()
    sourceTrackingScore = normalizeScore(sourceTrackingScore)

    return sourceTrackingScore

if __name__ == '__main__':
    
    resetDataset()
    createDataset()

    uid = 12
    prob = [0.1,0.2,0.3,0.4]

    addFinalResult(usersDSname, uid, prob)
    print(getScore(usersDSname, uid))


    uid = 12
    prob = [0.3,0.1,0.4,0.2]

    addFinalResult(usersDSname, uid, prob)
    print(getScore(usersDSname, uid))

    topic = "covid"
    prob = [0.05,0.3,0.15,0.5]

    addFinalResult(tagDSname, uid, prob)
    print(getScore(tagDSname, uid))

