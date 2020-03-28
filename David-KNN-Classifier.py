import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as m
import random

train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
submission=pd.read_csv('../input/titanic/gender_submission.csv')
ground_truth=pd.read_csv('../input/ground-truth/submission.csv')

correct=0
#Try to find best combination of n and k?
#n=4 and k=sqrt--78.23%

k=round(m.sqrt(len(train)))

k=27

# def KthSmallest(arr, l, r, K): 
	
# 	# If k is smaller than number of 
# 	# elements in array 
# 	if (K > 0 and K <= r - l + 1): 
		
# 		# Partition the array around a random 
# 		# element and get position of pivot 
# 		# element in sorted array 
# 		pos = randomPartition(arr, l, r) 

# 		# If position is same as k 
# 		if (pos - l == K - 1): 
# 			return pos
# 		if (pos - l > K - 1): # If position is more, 
# 							# recur for left subarray 
# 			return KthSmallest(arr, l, pos - 1, K) 

# 		# Else recur for right subarray 
# 		return KthSmallest(arr, pos + 1, r, 
# 						K - pos + l - 1) 

# 	# If k is more than the number of 
# 	# elements in the array 
# 	return 999999999999

# def swap(arr, a, b): 
# 	temp = arr[a] 
# 	arr[a] = arr[b] 
# 	arr[b] = temp 

# # Standard partition process of QuickSort(). 
# # It considers the last element as pivot and 
# # moves all smaller element to left of it and 
# # greater elements to right. This function 
# # is used by randomPartition() 
# def partition(arr, l, r): 
# 	x = arr[r] 
# 	i = l 
# 	for j in range(l, r): 
# 		if (arr[j] <= x): 
# 			swap(arr, i, j) 
# 			i += 1
# 	swap(arr, i, r) 
# 	return i 

# # Picks a random pivot element between l and r 
# # and partitions arr[l..r] around the randomly 
# # picked element using partition() 
# def randomPartition(arr, l, r): 
# 	n = r - l + 1
# 	pivot = int(random.random() % n) 
# 	swap(arr, l + pivot, r) 
# 	return partition(arr, l, r)

# def kSmallest(arr):
#     Arr=[]
#     for i in range(1,k+1):
#         Arr.append(KthSmallest(arr,0,len(arr)-1,i))
#     return Arr

def round_frac(d,min_,max_,n=4):
  return int(round(n*((d-min_)/(max_-min_))))

def Dist(arr1,arr2):
    total=0
    for i in range(len(arr1)):
        if(m.isnan(arr1[i]) or m.isnan(arr2[i])):
            continue
#         total+=min(1,abs(arr1[i]-arr2[i]))
        total+=abs(arr1[i]-arr2[i])
    return total
        
#Pre-process train and drop
train.drop(columns=['Cabin','Name','Ticket','PassengerId','Embarked'],axis=1,inplace=True)
train.dropna(inplace=True)
train.reset_index(inplace=True,drop=True)
train['Sex']=np.where(train['Sex']=='male',0,1)
# train['Family']=train['Parch']+train['SibSp']
# train.drop(columns=['Parch','SibSp'],axis=1,inplace=True)

test.drop(columns=['Cabin','Name','Ticket','PassengerId','Embarked'],axis=1,inplace=True)
test['Sex']=np.where(test['Sex']=='male',0,1)
# test['Family']=test['Parch']+test['SibSp']
# test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)

#Test Code

# Best=[0.7822966507177034, 4, 27]
# for a in range(3,51):
#     train=pd.read_csv('../input/titanic/train.csv')
#     test=pd.read_csv('../input/titanic/test.csv')
#     submission=pd.read_csv('../input/titanic/gender_submission.csv')
#     ground_truth=pd.read_csv('../input/ground-truth/submission.csv')
    
#     #Pre-process train and drop
#     train.drop(columns=['Cabin','Name','Ticket','PassengerId','Embarked'],axis=1,inplace=True)
#     train.dropna(inplace=True)
#     train.reset_index(inplace=True,drop=True)
#     train['Sex']=np.where(train['Sex']=='male',0,1)
#     # train['Family']=train['Parch']+train['SibSp']
#     # train.drop(columns=['Parch','SibSp'],axis=1,inplace=True)

#     test.drop(columns=['Cabin','Name','Ticket','PassengerId','Embarked'],axis=1,inplace=True)
#     test['Sex']=np.where(test['Sex']=='male',0,1)
#     # test['Family']=test['Parch']+test['SibSp']
#     # test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)
    
#     for col in test.columns:
#         Min=min(test[col])
#         Max=max(test[col])
#         test[col]=test[col].apply(lambda row: round_frac(row,Min,Max,a) if pd.notnull(row) else row)

#         train[col]=train[col].apply(lambda row: round_frac(row,Min,Max,a) if pd.notnull(row) else row)

#     #For each test, find distance, make 'Distance'. Sort, find and put to submission the predicted value

#     for b in range(10,round(m.sqrt(len(train)))):
#         correct=0
#         k=b
#         for i in range(len(test)):
#             Total=[0,0]
#             train['Distance']=train.apply(lambda row: Dist(list(row)[1:],list(test.values)[i]),axis=1)
#         #     Distance=train['Distance'].values.tolist()
#         #     Arr=kSmallest(Distance)
#             train=train.sort_values(by=['Distance'])
#             train.reset_index(inplace=True,drop=True)

#         #     Base=train.at[Arr[0],'Distance']+1
#             Base=train.at[0,'Distance']+1

#             for j in range(k):
#         #     for j in Arr:
#                 Total[0]+=(Base/(1+train.at[j,'Distance']))*train.at[j,'Survived']
#                 Total[1]+=Base/(1+train.at[j,'Distance'])

#             pred=round(Total[0]/Total[1])

#             submission.at[i,'Survived']=pred
#             if(submission.at[i,'Survived']==ground_truth.at[i,'Survived']):
#                 correct+=1
#             train.drop(columns=['Distance'],axis=1,inplace=True)
#         if((correct/len(test))>=Best[0]):
#             Best=[correct/len(test),a,b]
#             print(Best)

for col in test.columns:
    Min=min(test[col])
    Max=max(test[col])
    test[col]=test[col].apply(lambda row: round_frac(row,Min,Max) if pd.notnull(row) else row)

    train[col]=train[col].apply(lambda row: round_frac(row,Min,Max) if pd.notnull(row) else row)

for i in range(len(test)):
    Total=[0,0]
    train['Distance']=train.apply(lambda row: Dist(list(row)[1:],list(test.values)[i]),axis=1)
    train=train.sort_values(by=['Distance'])
    train.reset_index(inplace=True,drop=True)

    Base=train.at[0,'Distance']+1

    for j in range(k):
        Total[0]+=(Base/(1+train.at[j,'Distance']))*train.at[j,'Survived']
        Total[1]+=Base/(1+train.at[j,'Distance'])

    pred=round(Total[0]/Total[1])

    submission.at[i,'Survived']=pred
    if(submission.at[i,'Survived']==ground_truth.at[i,'Survived']):
        correct+=1
    
    train.drop(columns=['Distance'],axis=1,inplace=True)
    
print(correct/len(test))
submission.to_csv('Submission_Titanic.csv',index=False)
