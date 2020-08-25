import datetime
import argparse

parser = argparse.ArgumentParser(description='Generate Energy Consumption Database')
parser.add_argument('-samples', action='store', dest='samples',
                    help='Número de Muestras a Generar',type=int)
parser.add_argument('-prob', action='store', dest='prob',
                    help='Probabilidad de pago de las obligaciones asumida',type=float)

args = parser.parse_args()

#############################################################



import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import os
import warnings
from numpy.random import choice
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score,recall_score,cohen_kappa_score,f1_score,accuracy_score
import prince as pr ##por MCA




time_init=datetime.datetime.now()

data=pd.read_csv("DBCaribeSol.txt",sep="|",decimal=",")


data_consumo=data.groupby('Id_SubCategoria').ValorAnterior.agg({'mean','median','min','max','std'})
data_unidades=data.groupby('Id_SubCategoria').UnidadesAnt1.agg({'mean','median','min','max','std'})
index_est=data_consumo.index
dict_consumo=data_consumo.to_dict()
dict_unidades=data_unidades.to_dict()



##COSTO POR KWH supuesto para cada estrato
kwh_cost=dict({'1 - ESTRATO 1':[200,1],
               '2 - ESTRATO 2':[200,2],
               '3 - ESTRATO 3':[300,3],
               '4 - ESTRATO 4':[400,4],
               '5 - ESTRATO 5':[500,5],
               '6 - ESTRATO 6':[500,6],
               '11 - 220 Voltios.':[350,7]})

###SIMULACIÓN DE LA DATA:
def simula_cons_energy(N_SAMPLES=10000,p_pago=0.9):
  """N_SAMPLES: número de muestras aleatorias a generar
    p_pago: probabilidad de que las personas paguen su última obligación
  """

  p_pago=0.9
  props_est = data.groupby('Id_SubCategoria')['Id_SubCategoria'].count()/data.shape[0]
  props_loc= data.groupby('Localidad')['Localidad'].count()/data.shape[0]
  LOCALIDAD=choice(props_loc.index.values, N_SAMPLES, p=props_loc.values)
  ESTRATO=choice(props_est.index.values, N_SAMPLES, p=props_est.values)
  UNIDADES=[np.random.default_rng().normal(dict_unidades['mean'][i],dict_unidades['std'][i], 1)[0] for i in ESTRATO]
  med_cal=np.median(UNIDADES)
  print(f"Unidades de Consumo negativas se reemplazarán por {med_cal}")
  UNIDADES=np.where(np.array(UNIDADES)<=0,med_cal,UNIDADES)
  VALOR_CONS=[UNIDADES[i]*kwh_cost[ESTRATO[i]][0] for i in range(N_SAMPLES)]
  VALORDEF=np.quantile(VALOR_CONS,0.75)
  print(f"Considerando No pagos con valores de factura superiores a {VALORDEF}")
  NO_PAGO_ULTIMO=[0 if i<=VALORDEF else 1 for i in VALOR_CONS]
  data_out=pd.DataFrame({'localidad':LOCALIDAD,'estrato':ESTRATO,'valor_ant':VALOR_CONS,'unidades_ant':UNIDADES,'no_pago_ultimo':NO_PAGO_ULTIMO})
  return(data_out)

data_out=simula_cons_energy(args.samples,args.prob)

print("[INFO] Data chunk 1 ...done")

def cons_ultim_12(key_val,base_val):
  values=[]
  values.append(base_val)
  [values.append(np.abs((dict_unidades['std'][key_val]*1.02*choice([-1,1],1,p=[0.5,0.5])[0])+values[i])) for i in range(12)]
  return(values[::-1])
  
unidades_ultim_12=[cons_ultim_12(data_out.estrato.values[i],data_out.unidades_ant.values[i]) for i in range(len(data_out.estrato))];
unidades_ultim_12=np.reshape(unidades_ultim_12,((-1,13)))
unidades_ultim_12=pd.DataFrame(unidades_ultim_12,columns=["mes_t-"+str(12-i) for i in range(13)])

print("[INFO] Data chunk 2 ...done")

## modificar estrato a valor numérico
data_out['estrato']=data_out['estrato'].apply(lambda x: kwh_cost[x][1])
data_aux=data_out[['estrato','localidad','valor_ant']].copy()
data_aux.valor_ant=[(i-np.min(data_aux.valor_ant))/(np.max(data_aux.valor_ant)-np.min(data_aux.valor_ant)) for i in data_aux.valor_ant]
mca=pr.MCA(n_components=-1).fit_transform(data_aux.values)
gmm = GaussianMixture(n_components=4)
gmm.fit(mca)
labels = gmm.predict(mca)


### Modelo predictivo

### función para medir desempeño

def metrics(real,pred):
  kappa=cohen_kappa_score(real,pred)
  acc=accuracy_score(real,pred)
  f1=f1_score(real,pred)
  prec=precision_score(real,pred)
  recall=recall_score(real,pred)

  print (f" Accuracy:{acc:.4f} \n Precision: {prec:.4f} \n Recall: {recall:.4f} \n Kappa: {kappa:.4f} \n F1-Score: {f1:.4f} ")

##standardize by day (not by column)
def std_day(x,rang=13):
  """x: INPUT AS A NUMPY ARRAY 
    rang: columna hasta la cual de desea estandarizar
  """
  daily_data_Train=x[:,range(rang)].T
  scaled_features =StandardScaler()
  daily_dataSTD_Train=scaled_features.fit_transform(daily_data_Train).T
  return (daily_dataSTD_Train)

#X=X.reshape((-1,13,1))

X=unidades_ultim_12.values
X=std_day(X)
y=data_out.no_pago_ultimo.values


le=LabelEncoder()
loc_enc=le.fit_transform(data_out.localidad)
data_out.loc[:,'localidad']=loc_enc
data_out.head()

print(le.classes_)

X2=data_out.drop(columns=["no_pago_ultimo","unidades_ant"]).values
X=np.column_stack((X,X2))

### partitions for the model
X_1=X[:,range(13)].reshape((-1,13,1))
X_2=X[:,13::]

mod_loaded=tf.keras.models.load_model('stacking.h5')
print("====[INFO] Stacking model loaded===")

##exportar a csv
data_out['cluster']=labels ## se añade el cluster obtenido 
##usar el modelo recién cargado
data_out['no_paga_sig_mes']=np.argmax(mod_loaded.predict([X_1,X_2]),axis=1)
data_out.to_csv("datos_gen.csv",sep=";",decimal=",",index=False)

print("====[INFO] DATA OUT 1 ...DONE ====")

Z=unidades_ultim_12.values
Z=std_day(Z)

##usa los últimos 10 observados para pronosticar los 3 siguientes

newdata=(Z[:,3:]).reshape((-1,10,1))

### load model 
forec_load= tf.keras.models.load_model('forec.h5')
forecast_new=forec_load.predict(newdata)
values=unidades_ultim_12.apply(lambda x: [np.mean(x),np.std(x)],axis=1).to_dict()
##reconstruct original data

forecast_new=[(forecast_new[i]*values[i][1])+values[i][0] for i in range(len(forecast_new))]
forecast_new=pd.DataFrame(np.reshape(forecast_new,(-1,3)),columns=["mes_t-1","mes_t-2","mes_t-3"])
datos_consumo=pd.concat((unidades_ultim_12,forecast_new),axis=1)
datos_consumo.to_csv("consumos.csv",sep=";",decimal=",",index=False)

print("====[INFO] DATA OUT 2 ...DONE ====")

time_end=datetime.datetime.now()

print(f"====[INFO] All set, elapsed time {(time_end-time_init).total_seconds()/60} minutes ====")
