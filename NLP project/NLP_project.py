



#NLP (Natural Language Processing) KULLANARAK IMDB FÝLM 
# YORUMLARI KAGGLE DATA SETÝ ÜZERÝNDE SENTIMENT(DUYGU) ANALÝZÝ


#Bu projede NLP konsepti kullanarak  duygu analiz yazýlýmý gerçekleþtireceðim. 
#ingilizce IMDB sitesinden alýnan film yorumlarýndan oluþan bir veriseti kullanarak pozitif ve negatif yorumlarý birbirinden ayýran yapay zeka yapacaðým. 

# Bu projede iki þey kullanýyorum ; NLP konsepti ve Random Forest Classifier. Natural Language Processing yani
#3Doðal Dil Ýþleme olan **NLP**, dil bilimi, bilgisayar bilimi ve yapay zeka konularýnýn birleþtiði alandýr. Bilgisayarlarýn insan dillerini yani doðal dilleri nasýl anlayacaðýna 
#ve büyük doðal dil verilerini nasýl iþleyip analiz edeceðine dair çalýþmalarý kapsar. 
#Bu projede yapay zekanýn IMDB yorumlarýnýn olumlu mu olumsuz mu yorum olduðunu anlamasý için kullanýyorum.

#Peki Random Forest algoritmasý ne iþe yarar?
#birden çok karar aðacý üzerinden her bir karar aðacýný farklý bir gözlem örneði üzerinde eðiterek çeþitli modeller üretip, sýnýflandýrma oluþturmanýzý saðlamaktadýr



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



# Veri setlerimizi yüklüyoruz..
df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)

# Verimize bakalým
df.head()


len(df)



len(df["review"])



# stopwords'ü temizlemek için nltk kütüphanesinden stopwords kelime setini bilgisayarýmýza indirmemiz gerekiyor. 
# Bu iþlemi nltk ile yapýyoruz
nltk.download('stopwords')



# ## * * * * Veri Temizleme Ýþlemleri * * * *

# ### Öncelikle BeautifulSoup modülünü kullanarak HTML taglerini review cümlelerinden sileceðiz.
# Bu iþlemlerin nasýl yapýldýðýný açýklamak için önce örnek tek bir review seçip size nasýl yapýldýðýna bakalým:



sample_review= df.review[0]
sample_review


# HTML tagleri temizlendikten sonra..
sample_review = BeautifulSoup(sample_review).get_text()
sample_review




# noktalama iþaretleri ve sayýlardan temizliyoruz - regex kullanarak..
sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
sample_review



# küçük harfe dönüþtürüyoruz, makine öðrenim algoritmalarýmýzýn büyük harfle baþlayan kelimeleri farklý kelime olarak
# algýlamamasý için yapýyoruz bunu:
sample_review = sample_review.lower()
sample_review




# stopwords (yani the, is, are gibi kelimeler yapay zeka tarafýndan kullanýlmamasýný istiyoruz. Bunlar gramer kelimeri..)
# önce split ile kelimeleri bölüyoruz ve listeye dönüþtürüyoruz. amacýmýz stopwords kelimelerini çýkarmak..
sample_review = sample_review.split()



sample_review




len(sample_review)



# sample_review without stopwords
swords = set(stopwords.words("english"))                      # conversion into set for fast searching
sample_review = [w for w in sample_review if w not in swords]               
sample_review



len(sample_review)



# Temizleme iþlemini açýkladýktan sonra þimdi tüm dataframe'imiz içinde bulunan reviewleri döngü içinde topluca temizliyoruz
# bu amaçla önce bir fonksiyon oluþturuyoruz:


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
    # splitted paragraph'larý space ile birleþtiriyoruz return
    return(" ".join(review))



# training datamýzý yukardaki fonksiyon yardýmýyla temizliyoruz: 
# her 1000 review sonrasý bir satýr yazdýrarak review iþleminin durumunu görüyoruz..

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


# ### Bag of Words oluþturuyoruz !
# 
# Verilerimizi temizledik ancak yapay zekanýn çalýþmasý için bu metin tabanlý verileri sayýlara ve bag of words denilen matrise çevirmek gerekiyor. Ýþte bu amaçla sklearn içinde bulunan CountVectorizer aracýný kullanýyoruz:

# <IMG src="bag.jpg" width="900" height="900" >



# sklearn içinde bulunan countvectorizer fonksiyonunu kullanarak max 5000 kelimelik bag of words oluþturuyoruz...
vectorizer = CountVectorizer( max_features = 5000 )

# train verilerimizi feature vektöre matrisine çeviriyoruz
train_x = vectorizer.fit_transform(train_x)



train_x


# Bunu array'e dönüþtürüyoruz çünkü fit iþlemi için array istiyor..
train_x = train_x.toarray()
train_y = y_train



train_x.shape, train_y.shape


train_y


# ### Random Forest Modeli oluþturuyoruz ve fit ediyoruz



model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)



# ### Þimdi sýra test datamýzda..

# Test verilerimizi feature vektöre matrisine çeviriyoruz
# Yani ayný iþlemleri(bag of wordse dönüþtürme) tekrarlýyoruz bu sefer test datamýz için:
test_xx = vectorizer.transform(test_x)



test_xx



test_xx = test_xx.toarray()



test_xx.shape


# #### Prediction yapýyoruz..



test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)



print("Doðruluk oraný : % ", dogruluk * 100)
