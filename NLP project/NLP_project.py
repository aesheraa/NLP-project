



#NLP (Natural Language Processing) KULLANARAK IMDB F�LM 
# YORUMLARI KAGGLE DATA SET� �ZER�NDE SENTIMENT(DUYGU) ANAL�Z�


#Bu projede NLP konsepti kullanarak  duygu analiz yaz�l�m� ger�ekle�tirece�im. 
#ingilizce IMDB sitesinden al�nan film yorumlar�ndan olu�an bir veriseti kullanarak pozitif ve negatif yorumlar� birbirinden ay�ran yapay zeka yapaca��m. 

# Bu projede iki �ey kullan�yorum ; NLP konsepti ve Random Forest Classifier. Natural Language Processing yani
#3Do�al Dil ��leme olan�**NLP**, dil bilimi, bilgisayar bilimi ve yapay zeka konular�n�n birle�ti�i aland�r. Bilgisayarlar�n insan dillerini yani do�al dilleri nas�l anlayaca��na 
#ve b�y�k do�al dil verilerini nas�l i�leyip analiz edece�ine dair �al��malar� kapsar. 
#Bu projede yapay zekan�n IMDB yorumlar�n�n olumlu mu olumsuz mu yorum oldu�unu anlamas� i�in kullan�yorum.

#Peki Random Forest algoritmas� ne i�e yarar?
#birden �ok karar a�ac� �zerinden her bir karar a�ac�n� farkl� bir g�zlem �rne�i �zerinde e�iterek �e�itli modeller �retip, s�n�fland�rma olu�turman�z� sa�lamaktad�r



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords



# Veri setlerimizi y�kl�yoruz..
df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)

# Verimize bakal�m
df.head()


len(df)



len(df["review"])



# stopwords'� temizlemek i�in nltk k�t�phanesinden stopwords kelime setini bilgisayar�m�za indirmemiz gerekiyor. 
# Bu i�lemi nltk ile yap�yoruz
nltk.download('stopwords')



# ## * * * * Veri Temizleme ��lemleri * * * *

# ### �ncelikle BeautifulSoup mod�l�n� kullanarak HTML taglerini review c�mlelerinden silece�iz.
# Bu i�lemlerin nas�l yap�ld���n� a��klamak i�in �nce �rnek tek bir review se�ip size nas�l yap�ld���na bakal�m:



sample_review= df.review[0]
sample_review


# HTML tagleri temizlendikten sonra..
sample_review = BeautifulSoup(sample_review).get_text()
sample_review




# noktalama i�aretleri ve say�lardan temizliyoruz - regex kullanarak..
sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
sample_review



# k���k harfe d�n��t�r�yoruz, makine ��renim algoritmalar�m�z�n b�y�k harfle ba�layan kelimeleri farkl� kelime olarak
# alg�lamamas� i�in yap�yoruz bunu:
sample_review = sample_review.lower()
sample_review




# stopwords (yani the, is, are gibi kelimeler yapay zeka taraf�ndan kullan�lmamas�n� istiyoruz. Bunlar gramer kelimeri..)
# �nce split ile kelimeleri b�l�yoruz ve listeye d�n��t�r�yoruz. amac�m�z stopwords kelimelerini ��karmak..
sample_review = sample_review.split()



sample_review




len(sample_review)



# sample_review without stopwords
swords = set(stopwords.words("english"))                      # conversion into set for fast searching
sample_review = [w for w in sample_review if w not in swords]               
sample_review



len(sample_review)



# Temizleme i�lemini a��klad�ktan sonra �imdi t�m dataframe'imiz i�inde bulunan reviewleri d�ng� i�inde topluca temizliyoruz
# bu ama�la �nce bir fonksiyon olu�turuyoruz:


def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))                      # conversion into set for fast searching
    review = [w for w in review if w not in swords]               
    # splitted paragraph'lar� space ile birle�tiriyoruz return
    return(" ".join(review))



# training datam�z� yukardaki fonksiyon yard�m�yla temizliyoruz: 
# her 1000 review sonras� bir sat�r yazd�rarak review i�leminin durumunu g�r�yoruz..

train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))


# ### Train, test split..


x = train_x_tum
y = np.array(df["sentiment"])

# train test split
train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1)


# ### Bag of Words olu�turuyoruz !
# 
# Verilerimizi temizledik ancak yapay zekan�n �al��mas� i�in bu metin tabanl� verileri say�lara ve bag of words denilen matrise �evirmek gerekiyor. ��te bu ama�la sklearn i�inde bulunan CountVectorizer arac�n� kullan�yoruz:

# <IMG src="bag.jpg" width="900" height="900" >



# sklearn i�inde bulunan countvectorizer fonksiyonunu kullanarak max 5000 kelimelik bag of words olu�turuyoruz...
vectorizer = CountVectorizer( max_features = 5000 )

# train verilerimizi feature vekt�re matrisine �eviriyoruz
train_x = vectorizer.fit_transform(train_x)



train_x


# Bunu array'e d�n��t�r�yoruz ��nk� fit i�lemi i�in array istiyor..
train_x = train_x.toarray()
train_y = y_train



train_x.shape, train_y.shape


train_y


# ### Random Forest Modeli olu�turuyoruz ve fit ediyoruz



model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)



# ### �imdi s�ra test datam�zda..

# Test verilerimizi feature vekt�re matrisine �eviriyoruz
# Yani ayn� i�lemleri(bag of wordse d�n��t�rme) tekrarl�yoruz bu sefer test datam�z i�in:
test_xx = vectorizer.transform(test_x)



test_xx



test_xx = test_xx.toarray()



test_xx.shape


# #### Prediction yap�yoruz..



test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)



print("Do�ruluk oran� : % ", dogruluk * 100)
