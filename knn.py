import pandas as pd
from sklearn.preprocessing import MinMaxScaler #sadece veri normalizasyonu için sklearn kullandım
scaler = MinMaxScaler()
import time 
start=time.time()
i=0
j=1
y=0
deger=0
liste_ar=[]
son=0

###############################################################
df=pd.read_excel("Dry_Bean_Dataset.xlsx")
df.iloc[:,0:-1] = scaler.fit_transform(df.iloc[:,0:-1].to_numpy()) # verileri 0-1 arasında normalize ettik
df1=df.copy()
df1['index_col'] = df.index
###############################################################


train=df.sample(frac=0.7 ,random_state=61) #train sample

test=df.drop(train.index) #test sample

hesaplanacak=pd.DataFrame(train.iloc[0:1,:]) #sürekli değisecek olan classını tahmın edecegımız veri

y_sinif=hesaplanacak["Class"].values[0] #  var olan classımız budur

hesaplanacak=hesaplanacak.drop("Class",axis=1)   #hesaplayacagımızın sadece verılerını tutuyoruz classı cıkardık.

train_class=train.drop("Class",axis=1) # var olan tumm verıler classları ayır

train_class=train.drop(hesaplanacak.index)


len_test=hesaplanacak.shape[0]
len_train=train_class.shape[0]
len_columns=train_class.shape[1]


def hesapla_class(liste_ar,k_changed):
        
    sorted_lenght=sorted(liste_ar,key=lambda liste_ar:liste_ar[1], reverse=False)# uzaklıklara göre sıraliyoruz
    count=0
    clas_aga=[]
    while(1):
        if(count==k_changed):
            break
        indexler=(sorted_lenght[count][0])
        clas_aga.append(df1[df1["index_col"]==indexler]["Class"].values[0])
        count+=1
    y_head=max(clas_aga,key=clas_aga.count)
    liste_ar.clear()
    return y_head

count_true=0
count_false=0

def knn(df,train_class,hesaplanacak,y_sinif,k_changed):
    global count_true
    global count_false
    global len_columns
    global liste_ar
    global i
    global y
    global deger
    global j
    global son
    try:
        while(1):
            
            if(y==len_columns-1):
                sonuc=deger**0.5

                liste_ar.append((train_class.index.values[i],sonuc)) # indexleri ve uzaklıkları lıste_ar'ın içine atıyoruz ileride sıralamak için
                #print(liste_ar)
                sonuc=0
                deger=0
                i+=1
                y=0
            if(len_train<=i): # doğruluk oranı ıcın hesaplamaları bunun ıcınde anlık olarak tutuyorum
            
                tahmin=hesapla_class(liste_ar,k_changed)
                #print("Class",y_sinif)
                
                if(y_sinif==tahmin):
                    #print("{} GERCEK DEGERİM , {} TAHMİN DEGERİM ".format(y_sinif,tahmin))
                    count_true+=1
                else:
                    #print("{} GERCEK DEGERİM , {} TAHMİN DEGERİM ".format(y_sinif,tahmin))
                    count_false+=1

                j+=1
                hesaplanacak=pd.DataFrame(df.iloc[j:(j+1),:])
                y_sinif=hesaplanacak["Class"].values[0]
                hesaplanacak=hesaplanacak.drop("Class",axis=1) #class_ hesaplanacakın classını tuttuk
                train_class=df.drop("Class",axis=1) # var olan tumm verıler classları ayır
                train_class=df.drop(hesaplanacak.index)

                sonuc=0
                deger=0
                i=0
                y=0
                
                continue
            
            deger=deger+(train_class.iloc[i,y]-hesaplanacak.iloc[0,y])**2 #öklid hesabı için karekok harıc olan kısım
            y+=1

    except:
        
        son=(count_true/(count_true+count_false))*100
        #print("doğruluk orani",son)

try:
    k_changed=int(input("Bir 'K' parametresi giriniz "))
except:
    print("Hatali giris cikis yapiliyor")
    exit()  
knn(train,train_class,hesaplanacak,y_sinif,k_changed)
print("Train için doğruluk orani {:.2f} ".format(son))

print("----------")

i=0
j=1
y=0
deger=0
liste_ar=[]
count_true=0
count_false=0
son=0
hesaplanacak=pd.DataFrame(test.iloc[0:1,:])

y_sinif=hesaplanacak["Class"].values[0]

hesaplanacak=hesaplanacak.drop("Class",axis=1)   #class_ hesaplanacakın classını tuttuk
                                        
train_class=test.drop("Class",axis=1) # var olan tumm verıler classları ayır

train_class=test.drop(hesaplanacak.index)

len_test=hesaplanacak.shape[0]
len_train=train_class.shape[0]
len_columns=train_class.shape[1]

knn(test,train_class,hesaplanacak,y_sinif,k_changed)
print("Test için doğruluk orani {:.2f} ".format(son))

end=time.time()
sonucc=end-start
print("Hesaplama zamani",sonucc)