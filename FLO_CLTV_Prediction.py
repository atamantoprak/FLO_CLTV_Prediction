##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# FLO satış ve pazarlama faaliyetleri için roadmap
# belirlemek istemektedir. Şirketin orta uzun vadeli plan
# yapabilmesi için var olan müşterilerin gelecekte şirkete
# sağlayacakları potansiyel değerin tahmin edilmesi
# gerekmektedir.

# Veri Seti Hikayesi

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.


# Değişkenler

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def data_prep(dataframe):
    df["omni_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["omni_value_total_ever"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    return dataframe

#########################
# Verinin Okunması
#########################

df_ = pd.read_csv("FLO_CLTV_Prediction/flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#########################
# Veri Ön İşleme
#########################
# Aykırı değerleri baskılama işlemi

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# Analiz tarihi tanımlama
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

#  Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturma işlemi.
data_prep(df)
df.info()

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((today_date - df["last_order_date"]).astype("timedelta64[D]"))/ 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]"))/7
cltv_df["frequency"] = df["omni_num_total_ever"]
cltv_df["monetary_cltv_avg"] =  df["omni_value_total_ever"] / cltv_df["frequency"]

cltv_df.head()


##############################################################
#  BG-NBD Modelinin Kurulması
##############################################################
# BG-NBD modelinin kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin edip yeni bitr değişken olarak cltv dataframe'ine ekleme işlemi.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin edip yeni bitr değişken olarak cltv dataframe'ine ekleme işlemi.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

# Tahmin Sonuçlarının Değerlendirilmesi
plot_period_transactions(bgf)
plt.show()


##############################################################
# GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()
##############################################################
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################
# 6 aylık cltv hesapladık
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

##############################################################
# CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_df.sort_values("cltv",ascending=False).head(20)

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head(20)


##############################################################
# Çalışmanın Fonksiyonlaştırılması
##############################################################

def create_cltv_df(dataframe):

    # Veri Ön İşleme
    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")

    today_date = dt.datetime(2021, 6, 1)
    data_prep(df)

    # Lifetime Veri Yapısının Hazırlanması
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((today_date - df["last_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["frequency"] = df["omni_num_total_ever"]
    cltv_df["monetary_cltv_avg"] = df["omni_value_total_ever"] / cltv_df["frequency"]

    # BG-NBD modelinin kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    #  BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # 5. CLTV'ye Göre Segmentlerin Oluşturulması
    cltv_df.sort_values("cltv", ascending=False).head(20)
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

a = create_cltv_df(df)
a.head()