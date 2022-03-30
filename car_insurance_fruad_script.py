from numpy import random
from random import randrange
import numpy as np
from datetime import datetime, timedelta
import pandas as pds
from configparser import ConfigParser
import time
import os
import xlsxwriter
import csv
read_config = ConfigParser()
workbook = xlsxwriter.Workbook('C:/Users/AbdulSamadKH/Downloads/replace1.xlsx')
#workbook = xlsxwriter.Workbook('replace1.xlsx')
worksheet = workbook.add_worksheet('falg_0')
read_config.read("C:/Users/AbdulSamadKH/Downloads/testing1.ini")
rang_e= read_config.get("dynamic_colns","range")
ran_ge=int(rang_e)
Id= read_config.get("dynamic_colns","Id")
id=int(Id)
worksheet.write(0, 0, "ID")
for i in range(ran_ge):
    id=id+1
    worksheet.write(i + 1, 0, id)
KIDSDRIV= read_config.get("dynamic_colns","KIDSDRIV")
kids_driv=KIDSDRIV.strip().replace(' ', '').split(',')
worksheet.write(0, 1, "KIDSDRIV")
for i in range(ran_ge):
    random_kids_driv = random.randint(kids_driv[0],kids_driv[1])
    worksheet.write(i+1, 1, random_kids_driv)
a_ge = read_config.get("dynamic_colns","age")
age=a_ge.strip().replace(' ', '').split(',')
worksheet.write(0, 2, "age")
worksheet.write(0, 3, "Birth")
month = read_config.get("dynamic_colns","month")
mon_th=month.strip().replace(' ', '').split(',')
for i in range(ran_ge):
    random_age = random.randint(age[0], age[1])
    worksheet.write(i+1, 2, random_age)
    bir_th=2022-random_age
    bir_th=str(bir_th)
    random_month=random.choice([mon_th[0],mon_th[1],mon_th[2],
                                mon_th[3],mon_th[4],mon_th[5],
                                mon_th[6],mon_th[7],mon_th[8],
                                mon_th[9],mon_th[10],mon_th[11]])
    month1=random_month+'/'+str(1)+'/'+bir_th
   # print(type(month1))
    month1=str(month1)
    worksheet.write(i + 1, 3, month1)
HOMEKIDS= read_config.get("dynamic_colns","HOMEKIDS")
HOMEKIDS=HOMEKIDS.strip().replace(' ', '').split(',')
worksheet.write(0, 4, "HOMEKIDS")
for i in range(ran_ge):
    random_HOMEKIDS= random.randint(HOMEKIDS[0],HOMEKIDS[1])
    worksheet.write(i+1, 4, random_HOMEKIDS)
YOJ= read_config.get("dynamic_colns","YOJ")
YOJ=YOJ.strip().replace(' ', '').split(',')
worksheet.write(0, 5, "YOJ")
for i in range(ran_ge):
    random_YOJ= random.randint(YOJ[0],YOJ[1])
    worksheet.write(i+1, 5, random_YOJ)
income= read_config.get("dynamic_colns","income")
income=income.strip().replace(' ', '').split(',')
worksheet.write(0, 6, "income")
for i in range(ran_ge):
    random_income= random.randint(income[0],income[1])
    random_income=str(random_income)
    random_inco="$"+random_income
    worksheet.write(i+1, 6, random_inco)
PARENT1= read_config.get("dynamic_colns","PARENT1")
PARENT1=PARENT1.strip().replace(' ', '').split(',')
worksheet.write(0, 7, "PARENT1")
for i in range(ran_ge):
    random_PARENT1 = random.choice([PARENT1[0],PARENT1[1]])
    worksheet.write(i+1, 7, random_PARENT1)
HOME_VAL= read_config.get("dynamic_colns","HOME_VAL")
HOME_VAL=HOME_VAL.strip().replace(' ', '').split(',')
worksheet.write(0, 8, "HOME_VAL")
for i in range(ran_ge):
    random_HOME_VAL= random.randint(HOME_VAL[0],HOME_VAL[1])
    random_HOME_VAL=str(random_HOME_VAL)
    random_home_val="$"+random_HOME_VAL
    worksheet.write(i+1, 8, random_home_val)
MSTATUS= read_config.get("dynamic_colns","MSTATUS")
MSTATUS=MSTATUS.strip().replace(' ', '').split(',')
worksheet.write(0, 9, "MSTATUS")
for i in range(ran_ge):
    random_MSTATUS = random.choice([MSTATUS[0],MSTATUS[1]])
    worksheet.write(i+1, 9, random_MSTATUS)
Gender= read_config.get("dynamic_colns","Gender")
Gender=Gender.strip().replace(' ', '').split(',')
worksheet.write(0, 10, "GENDER")
for i in range(ran_ge):
    random_Gender = random.choice([Gender[0],Gender[1]])
    worksheet.write(i+1, 10, random_Gender)
Education= read_config.get("dynamic_colns","Education")
Education=Education.strip().replace(' ', '').split(',')
worksheet.write(0, 11, "Education")
for i in range(ran_ge):
    random_Education = random.choice([Education[0],Education[1],Education[2],
                                   Education[3],Education[4]])
    worksheet.write(i+1, 11, random_Education)
OCCUPATION= read_config.get("dynamic_colns","OCCUPATION")
OCCUPATION=OCCUPATION.strip().replace(' ', '').split(',')
worksheet.write(0, 12, "OCCUPATION")
for i in range(ran_ge):
    random_OCCUPATION = random.choice([OCCUPATION[0],OCCUPATION[1],OCCUPATION[2],
                                   OCCUPATION[3],OCCUPATION[4],OCCUPATION[5],OCCUPATION[6],OCCUPATION[7]])
    worksheet.write(i+1, 12, random_OCCUPATION)
TRAVTIME= read_config.get("dynamic_colns","TRAVTIME")
TRAVTIME=TRAVTIME.strip().replace(' ', '').split(',')
worksheet.write(0, 13, "TRAVTIME")
for i in range(ran_ge):
    random_TRAVTIME= random.randint(TRAVTIME[0],TRAVTIME[1])
    worksheet.write(i + 1, 13, random_TRAVTIME)
CAR_USE= read_config.get("dynamic_colns","CAR_USE")
CAR_USE=CAR_USE.strip().replace(' ', '').split(',')
worksheet.write(0, 14, "CAR_USE")
for i in range(ran_ge):
    random_CAR_USE = random.choice([CAR_USE[0],CAR_USE[1]])
    worksheet.write(i+1, 14, random_CAR_USE)
BLUEBOOK= read_config.get("dynamic_colns","BLUEBOOK")
BLUEBOOK=BLUEBOOK.strip().replace(' ', '').split(',')
worksheet.write(0, 15, "BLUEBOOK")
for i in range(ran_ge):
    random_BLUEBOOK= random.randint(BLUEBOOK[0],BLUEBOOK[1])
    random_BLUEBOOK=str(random_BLUEBOOK)
    random_BLUEBOOK="$"+random_BLUEBOOK
    worksheet.write(i+1, 15, random_BLUEBOOK)
TIF= read_config.get("dynamic_colns","TIF")
TIF=TIF.strip().replace(' ', '').split(',')
worksheet.write(0, 16, "TIF")
for i in range(ran_ge):
    random_tif= random.randint(TIF[0],TIF[1])
    worksheet.write(i + 1, 16, random_tif)
CAR_TYPE= read_config.get("dynamic_colns","CAR_TYPE")
CAR_TYPE=CAR_TYPE.strip().replace(' ', '').split(',')
worksheet.write(0, 17, "CAR_TYPE")
for i in range(ran_ge):
    random_CAR_TYPE = random.choice([CAR_TYPE[0],CAR_TYPE[1],CAR_TYPE[2],
                                   CAR_TYPE[3],CAR_TYPE[4]])
    worksheet.write(i+1, 17, random_CAR_TYPE)
RED_CAR= read_config.get("dynamic_colns","RED_CAR")
RED_CAR=RED_CAR.strip().replace(' ', '').split(',')
worksheet.write(0, 18, "RED_CAR")
for i in range(ran_ge):
    random_RED_CAR = random.choice([RED_CAR[0],RED_CAR[1]])
    worksheet.write(i+1, 18, random_RED_CAR)
OLDCLAIM= read_config.get("dynamic_colns","OLDCLAIM")
OLDCLAIM=OLDCLAIM.strip().replace(' ', '').split(',')
worksheet.write(0, 19, "OLDCLAIM")
for i in range(ran_ge):
    random_OLDCLAIM= random.randint(OLDCLAIM[0],OLDCLAIM[1])
    random_OLDCLAIM=str(random_OLDCLAIM)
    random_OLDCLAIM="$"+random_OLDCLAIM
    worksheet.write(i+1, 19, random_OLDCLAIM)
CLM_FREQ= read_config.get("dynamic_colns","CLM_FREQ")
CLM_FREQ=CLM_FREQ.strip().replace(' ', '').split(',')
worksheet.write(0, 20, "CLM_FREQ")
for i in range(ran_ge):
    random_CLM_FREQ= random.randint(CLM_FREQ[0],CLM_FREQ[1])
    worksheet.write(i + 1, 20, random_CLM_FREQ)
REVOKED= read_config.get("dynamic_colns","REVOKED")
REVOKED=REVOKED.strip().replace(' ', '').split(',')
worksheet.write(0, 21, "REVOKED")
for i in range(ran_ge):
    random_REVOKED= random.choice([REVOKED[0],REVOKED[1]])
    worksheet.write(i+1, 21, random_REVOKED)
MVR_PTS= read_config.get("dynamic_colns","MVR_PTS")
MVR_PTS=MVR_PTS.strip().replace(' ', '').split(',')
worksheet.write(0, 22, "MVR_PTS")
for i in range(ran_ge):
    random_MVR_PTS= random.randint(MVR_PTS[0],MVR_PTS[1])
    worksheet.write(i+1, 22, random_MVR_PTS)
CLM_AMT= read_config.get("dynamic_colns","CLM_AMT")
CLM_AMT=CLM_AMT.strip().replace(' ', '').split(',')
worksheet.write(0, 23, "CLM_AMT")
for i in range(ran_ge):
    random_CLM_AMT= random.randint(CLM_AMT[0],CLM_AMT[1])
    random_CLM_AMT=str(random_CLM_AMT)
    random_CLM_AMT="$"+random_CLM_AMT
    worksheet.write(i+1, 23, random_CLM_AMT)
CAR_AGE= read_config.get("dynamic_colns","CAR_AGE")
CAR_AGE=CAR_AGE.strip().replace(' ', '').split(',')
worksheet.write(0, 24, "CAR_AGE")
for i in range(ran_ge):
    random_CAR_AGE= random.randint(CAR_AGE[0],CAR_AGE[1])
    worksheet.write(i+1, 24, random_CAR_AGE)
CLAIM_FLAG= read_config.get("dynamic_colns","CLAIM_FLAG")
#CLAIM_FLAG=CLAIM_FLAG.strip().replace(' ', '').split(',')
CLAIM_FLAG=str(CLAIM_FLAG)
worksheet.write(0, 25, "CLAIM_FLAG")
for i in range(ran_ge):
    random_CLAIM_FLAG=CLAIM_FLAG
    worksheet.write(i+1, 25, random_CLAIM_FLAG)
URBANICITY= read_config.get("dynamic_colns","URBANICITY")
URBANICITY=URBANICITY.strip().replace(' ', '').split(',')
worksheet.write(0, 26, "URBANICITY")
for i in range(ran_ge):
    random_URBANICITY = random.choice([URBANICITY[0],URBANICITY[1]])
    worksheet.write(i+1, 26, random_URBANICITY)
rang_e1= read_config.get("dynamic_colns","range1")
ran_ge1=int(rang_e1)
Id= read_config.get("dynamic_colns","Id")
id=int(Id)
worksheet = workbook.add_worksheet('falg_1')
worksheet.write(0, 0, "ID")
for i in range(ran_ge1):
    id=id+1
    worksheet.write(i + 1, 0, id)
KIDSDRIV= read_config.get("dynamic_colns","KIDSDRIV")
kids_driv=KIDSDRIV.strip().replace(' ', '').split(',')
worksheet.write(0, 1, "KIDSDRIV")
for i in range(ran_ge1):
    random_kids_driv = random.randint(kids_driv[0],kids_driv[1])
    worksheet.write(i+1, 1, random_kids_driv)
a_ge = read_config.get("dynamic_colns","age")
age=a_ge.strip().replace(' ', '').split(',')
worksheet.write(0, 2, "age")
worksheet.write(0, 3, "Birth")
month = read_config.get("dynamic_colns","month")
mon_th=month.strip().replace(' ', '').split(',')
for i in range(ran_ge1):
    random_age = random.randint(age[0], age[1])
    worksheet.write(i+1, 2, random_age)
    bir_th=2022-random_age
    bir_th=str(bir_th)
    random_month=random.choice([mon_th[0],mon_th[1],mon_th[2],
                                mon_th[3],mon_th[4],mon_th[5],
                                mon_th[6],mon_th[7],mon_th[8],
                                mon_th[9],mon_th[10],mon_th[11]])
    month1=random_month+'/'+str(1)+'/'+bir_th
   # print(type(month1))
    month1=str(month1)
    worksheet.write(i + 1, 3, month1)
HOMEKIDS= read_config.get("dynamic_colns","HOMEKIDS")
HOMEKIDS=HOMEKIDS.strip().replace(' ', '').split(',')
worksheet.write(0, 4, "HOMEKIDS")
for i in range(ran_ge1):
    random_HOMEKIDS= random.randint(HOMEKIDS[0],HOMEKIDS[1])
    worksheet.write(i+1, 4, random_HOMEKIDS)
YOJ= read_config.get("dynamic_colns","YOJ")
YOJ=YOJ.strip().replace(' ', '').split(',')
worksheet.write(0, 5, "YOJ")
for i in range(ran_ge1):
    random_YOJ= random.randint(YOJ[0],YOJ[1])
    worksheet.write(i+1, 5, random_YOJ)
income= read_config.get("dynamic_colns","income")
income=income.strip().replace(' ', '').split(',')
worksheet.write(0, 6, "income")
for i in range(ran_ge1):
    random_income= random.randint(income[0],income[1])
    random_income=str(random_income)
    random_inco="$"+random_income
    worksheet.write(i+1, 6, random_inco)
PARENT1= read_config.get("dynamic_colns","PARENT1")
PARENT1=PARENT1.strip().replace(' ', '').split(',')
worksheet.write(0, 7, "PARENT1")
for i in range(ran_ge1):
    random_PARENT1 = random.choice([PARENT1[0],PARENT1[1]])
    worksheet.write(i+1, 7, random_PARENT1)
HOME_VAL= read_config.get("dynamic_colns","HOME_VAL")
HOME_VAL=HOME_VAL.strip().replace(' ', '').split(',')
worksheet.write(0, 8, "HOME_VAL")
for i in range(ran_ge1):
    random_HOME_VAL= random.randint(HOME_VAL[0],HOME_VAL[1])
    random_HOME_VAL=str(random_HOME_VAL)
    random_home_val="$"+random_HOME_VAL
    worksheet.write(i+1, 8, random_home_val)
MSTATUS= read_config.get("dynamic_colns","MSTATUS")
MSTATUS=MSTATUS.strip().replace(' ', '').split(',')
worksheet.write(0, 9, "MSTATUS")
for i in range(ran_ge1):
    random_MSTATUS = random.choice([MSTATUS[0],MSTATUS[1]])
    worksheet.write(i+1, 9, random_MSTATUS)
Gender= read_config.get("dynamic_colns","Gender")
Gender=Gender.strip().replace(' ', '').split(',')
worksheet.write(0, 10, "GENDER")
for i in range(ran_ge1):
    random_Gender = random.choice([Gender[0],Gender[1]])
    worksheet.write(i+1, 10, random_Gender)
Education= read_config.get("dynamic_colns","Education")
Education=Education.strip().replace(' ', '').split(',')
worksheet.write(0, 11, "Education")
for i in range(ran_ge1):
    random_Education = random.choice([Education[0],Education[1],Education[2],
                                   Education[3],Education[4]])
    worksheet.write(i+1, 11, random_Education)
OCCUPATION= read_config.get("dynamic_colns","OCCUPATION")
OCCUPATION=OCCUPATION.strip().replace(' ', '').split(',')
worksheet.write(0, 12, "OCCUPATION")
for i in range(ran_ge1):
    random_OCCUPATION = random.choice([OCCUPATION[0],OCCUPATION[1],OCCUPATION[2],
                                   OCCUPATION[3],OCCUPATION[4],OCCUPATION[5],OCCUPATION[6],OCCUPATION[7]])
    worksheet.write(i+1, 12, random_OCCUPATION)
TRAVTIME= read_config.get("dynamic_colns","TRAVTIME")
TRAVTIME=TRAVTIME.strip().replace(' ', '').split(',')
worksheet.write(0, 13, "TRAVTIME")
for i in range(ran_ge1):
    random_TRAVTIME= random.randint(TRAVTIME[0],TRAVTIME[1])
    worksheet.write(i + 1, 13, random_TRAVTIME)
CAR_USE= read_config.get("dynamic_colns","CAR_USE")
CAR_USE=CAR_USE.strip().replace(' ', '').split(',')
worksheet.write(0, 14, "CAR_USE")
for i in range(ran_ge1):
    random_CAR_USE = random.choice([CAR_USE[0],CAR_USE[1]])
    worksheet.write(i+1, 14, random_CAR_USE)
BLUEBOOK= read_config.get("dynamic_colns","BLUEBOOK")
BLUEBOOK=BLUEBOOK.strip().replace(' ', '').split(',')
worksheet.write(0, 15, "BLUEBOOK")
for i in range(ran_ge1):
    random_BLUEBOOK= random.randint(BLUEBOOK[0],BLUEBOOK[1])
    random_BLUEBOOK=str(random_BLUEBOOK)
    random_BLUEBOOK="$"+random_BLUEBOOK
    worksheet.write(i+1, 15, random_BLUEBOOK)
TIF= read_config.get("dynamic_colns","TIF")
TIF=TIF.strip().replace(' ', '').split(',')
worksheet.write(0, 16, "TIF")
for i in range(ran_ge1):
    random_tif= random.randint(TIF[0],TIF[1])
    worksheet.write(i + 1, 16, random_tif)
CAR_TYPE= read_config.get("dynamic_colns","CAR_TYPE")
CAR_TYPE=CAR_TYPE.strip().replace(' ', '').split(',')
worksheet.write(0, 17, "CAR_TYPE")
for i in range(ran_ge1):
    random_CAR_TYPE = random.choice([CAR_TYPE[0],CAR_TYPE[1],CAR_TYPE[2],
                                   CAR_TYPE[3],CAR_TYPE[4]])
    worksheet.write(i+1, 17, random_CAR_TYPE)
RED_CAR= read_config.get("dynamic_colns","RED_CAR")
RED_CAR=RED_CAR.strip().replace(' ', '').split(',')
worksheet.write(0, 18, "RED_CAR")
for i in range(ran_ge1):
    random_RED_CAR = random.choice([RED_CAR[0],RED_CAR[1]])
    worksheet.write(i+1, 18, random_RED_CAR)
OLDCLAIM= read_config.get("dynamic_colns","OLDCLAIM")
OLDCLAIM=OLDCLAIM.strip().replace(' ', '').split(',')
worksheet.write(0, 19, "OLDCLAIM")
for i in range(ran_ge1):
    random_OLDCLAIM= random.randint(OLDCLAIM[0],OLDCLAIM[1])
    random_OLDCLAIM=str(random_OLDCLAIM)
    random_OLDCLAIM="$"+random_OLDCLAIM
    worksheet.write(i+1, 19, random_OLDCLAIM)
CLM_FREQ= read_config.get("dynamic_colns","CLM_FREQ")
CLM_FREQ=CLM_FREQ.strip().replace(' ', '').split(',')
worksheet.write(0, 20, "CLM_FREQ")
for i in range(ran_ge1):
    random_CLM_FREQ= random.randint(CLM_FREQ[0],CLM_FREQ[1])
    worksheet.write(i + 1, 20, random_CLM_FREQ)
REVOKED= read_config.get("dynamic_colns","REVOKED")
REVOKED=REVOKED.strip().replace(' ', '').split(',')
worksheet.write(0, 21, "REVOKED")
for i in range(ran_ge1):
    random_REVOKED= random.choice([REVOKED[0],REVOKED[1]])
    worksheet.write(i+1, 21, random_REVOKED)
MVR_PTS= read_config.get("dynamic_colns","MVR_PTS")
MVR_PTS=MVR_PTS.strip().replace(' ', '').split(',')
worksheet.write(0, 22, "MVR_PTS")
for i in range(ran_ge1):
    random_MVR_PTS= random.randint(MVR_PTS[0],MVR_PTS[1])
    worksheet.write(i+1, 22, random_MVR_PTS)
CLM_AMT= read_config.get("dynamic_colns","CLM_AMT")
CLM_AMT=CLM_AMT.strip().replace(' ', '').split(',')
worksheet.write(0, 23, "CLM_AMT")
for i in range(ran_ge1):
    random_CLM_AMT= random.randint(CLM_AMT[0],CLM_AMT[1])
    random_CLM_AMT=str(random_CLM_AMT)
    random_CLM_AMT="$"+random_CLM_AMT
    worksheet.write(i+1, 23, random_CLM_AMT)
CAR_AGE= read_config.get("dynamic_colns","CAR_AGE")
CAR_AGE=CAR_AGE.strip().replace(' ', '').split(',')
worksheet.write(0, 24, "CAR_AGE")
for i in range(ran_ge1):
    random_CAR_AGE= random.randint(CAR_AGE[0],CAR_AGE[1])
    worksheet.write(i+1, 24, random_CAR_AGE)
CLAIM_FLAG1= read_config.get("dynamic_colns","CLAIM_FLAG1")
#CLAIM_FLAG=CLAIM_FLAG.strip().replace(' ', '').split(',')
CLAIM_FLAG1=str(CLAIM_FLAG1)
worksheet.write(0, 25, "CLAIM_FLAG")
for i in range(ran_ge1):
    random_CLAIM_FLAG1=CLAIM_FLAG1
    worksheet.write(i+1, 25, random_CLAIM_FLAG1)
URBANICITY= read_config.get("dynamic_colns","URBANICITY")
URBANICITY=URBANICITY.strip().replace(' ', '').split(',')
worksheet.write(0, 26, "URBANICITY")
for i in range(ran_ge1):
    random_URBANICITY = random.choice([URBANICITY[0],URBANICITY[1]])
    worksheet.write(i+1, 26, random_URBANICITY)
#workbook = xlsxwriter.Workbook('C:/Users/AbdulSamadKH/Downloads/replace1.xlsx')
workbook.close()
file =('C:/Users/AbdulSamadKH/Downloads/replace1.xlsx')
newData = pds.read_excel(file)
sheet1 = pds.read_excel(file,
                        sheet_name=0,
                        index_col=0)
sheet2 = pds.read_excel(file,
                        sheet_name=1,
                        index_col=0)

newData = pds.concat([sheet1, sheet2])
df = pds.DataFrame(newData)
df = df.sample(frac = 1)
df.to_excel('C:/Users/AbdulSamadKH/Downloads/Output File.xlsx')