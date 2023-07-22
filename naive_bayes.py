import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


df = pd.read_excel("Dry_Bean_Dataset.xlsx")
df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy()) #verileri normalize ettik

train = df.sample(frac = 0.7, random_state = 42) # yuzde 70 30 ayrım
test = df.drop(train.index)
y_train = train["Class"]
x_train = train.drop("Class", axis = 1)

y_test = test["Class"]
x_test = test.drop("Class", axis = 1)

#####
classes=[]
flag=0
for i in train["Class"]: #classlarımızı depoluyoruz

	for b in classes:
		if(i==b):
			flag=1
			break
		else:
			flag=0

	leng=len(classes)
	if(leng==7):
		break
	if(flag!=1):
		classes.append(i)
		flag=0


df_new=pd.DataFrame()
df_new_var=pd.DataFrame()
df_new_prior=pd.DataFrame()

for i in range(0,len(classes)): # ortalama standart sapma ve ilk olasılıkları hesaplayarak dataframe'e çeviriyoruz.
    
	df_new_temp=pd.DataFrame(train[train['Class'] == classes[i]].mean(numeric_only=True),columns=[classes[i]]).T
	temp=df_new_temp
	df_new=pd.concat([df_new, temp], ignore_index=False)

	df_new_temp_var=pd.DataFrame(train[train['Class'] == classes[i]].var(numeric_only=True)**0.5,columns=[classes[i]]).T
	temp_var=df_new_temp_var
	df_new_var=pd.concat([df_new_var, temp_var], ignore_index=False)

	df_new_temp_prior=(pd.DataFrame(train[train['Class'] == classes[i]].count(numeric_only=True)/len(train["Class"]==classes[i]),columns=[classes[i]])).iloc[-1]
	temp_prior=df_new_temp_prior
	df_new_prior=pd.concat([df_new_prior, temp_prior], ignore_index=False)

means=df_new
stdev=df_new_var
prior=df_new_prior[0]
#print(means)

def sayisal_hesap(x, mean, stdev):
	#sayısal değerlerde olasılık hesabı denklemi
	exponent = math.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent 


def Predict(x_data):
	Predictions = []
	
	for i in x_data.index: # bir verimizi secıyoruz
	   
		ClassLikelihood = []
		instance = x_data.loc[i]
		
		for j in classes: #j classların içine giriyor
			
			FeatureLikelihoods = []
			FeatureLikelihoods.append(prior[j]) #ilk olasılığı ekliyoruz
			
			for k in x_train.columns: #columnslara girerek hesap yapacagız
	
				data = instance[k]
	
				mean = means[k].loc[j]  #ortalamayı alıyoruz
				variance = stdev[k].loc[j] #varyansı alıyoruz
				
				Likelihood = sayisal_hesap(data, mean, variance) # sayısal değerlerde hesaplama fonksıyonu
				
				if(Likelihood==0):

					Likelihood = 0.05 # sabit olarak almak istedim
									  # 0.05 means olarak dusunerekten 
									  # hesaplamayı bozmayacagını tahmın edıyorum
									  
				FeatureLikelihoods.append(Likelihood)
			ClassLikelihood.append(np.prod(FeatureLikelihoods))	
	    #max olasılıgı seçmek ıcın yazılan kod parcacıgı
		max_prob=max(ClassLikelihood)
		count=0
		for i in ClassLikelihood:
			if(i==max_prob):
				max_prob_index=count
				break
			count+=1
		
		Prediction = classes[max_prob_index]

		Predictions.append(Prediction)
		
	return Predictions

def Accuracy(y, y_head):#doğruluk hesabını yaptıgımız bölüm
	y = list(y)
	y_head = list(y_head)
	score = 0

	for i, j in zip(y, y_head):
		if i == j:
			score += 1

	return score / len(y)        

PredictTrain = Predict(x_train)
PredictTest = Predict(x_test)

print("Train sonuç ",round(Accuracy(y_train, PredictTrain), 5))
print("Test sonuç ",round(Accuracy(y_test, PredictTest), 5))

