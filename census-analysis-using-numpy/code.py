# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data=np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data,new_record))


# --------------
#Code starts here
age = census[:,0]
max_age = max(age)
min_age = min(age)
age_mean = np.mean(age)
age_std = np.std(age)


# --------------
#Code starts here
race_0 = census[census[:,2]==0]
race_1 = census[census[:,2]==1]
race_2 = census[census[:,2]==2]
race_3 = census[census[:,2]==3]
race_4 = census[census[:,2]==4]

len_0 = len(race_0[:,2])
len_1 = len(race_1[:,2])
len_2 = len(race_2[:,2])
len_3 = len(race_3[:,2])
len_4 = len(race_4[:,2])

lengths = {}
lengths['len_0']=len_0
lengths['len_1']=len_1
lengths['len_2']=len_2
lengths['len_3']=len_3
lengths['len_4']=len_4

min_race=min(lengths.items(), key=lambda x:x[1])[0]

if min_race == 'len_0':
    minority_race = 0
elif min_race == 'len_1':
    minority_race = 1
elif min_race == 'len_2':
    minority_race = 2
elif min_race == 'len_3':
    minority_race = 3
else:
     minority_race = 4

print(minority_race)




# --------------
#Code starts here
senior_citizens = census[census[:,0]>60]
working_hours_sum = sum(senior_citizens[:,6])
senior_citizens_len = len(senior_citizens[:,6])
avg_working_hours = working_hours_sum/senior_citizens_len
print(avg_working_hours)


# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]

avg_pay_high = np.mean(high[:,7])
avg_pay_low = np.mean(low[:,7])




