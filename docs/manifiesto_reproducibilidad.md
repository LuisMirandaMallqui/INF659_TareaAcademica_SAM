# Manifiesto de reproducibilidad: datos reales usados

Dataset: Sen1Floods11 v1.1, split de validacion oficial.
CSV de split (89 filas reales): https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv

Regla de seleccion: primera fila de cada region, en el orden en que aparecen las regiones en el CSV original, recorriendo Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA. No hubo seleccion manual de chips favorables.

Chips descargados (10 de 89; 9 evaluables, Ghana_5079 excluido por 100% nodata de etiqueta):

| Chip | URL imagen S1Hand | URL mascara LabelHand |
|---|---|---|
| Ghana_5079 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Ghana_5079_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_5079_LabelHand.tif |
| India_1050276 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/India_1050276_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/India_1050276_LabelHand.tif |
| Mekong_1149855 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Mekong_1149855_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Mekong_1149855_LabelHand.tif |
| Nigeria_31096 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Nigeria_31096_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Nigeria_31096_LabelHand.tif |
| Pakistan_43105 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Pakistan_43105_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Pakistan_43105_LabelHand.tif |
| Paraguay_305760 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Paraguay_305760_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Paraguay_305760_LabelHand.tif |
| Somalia_12849 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Somalia_12849_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Somalia_12849_LabelHand.tif |
| Spain_1199913 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Spain_1199913_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Spain_1199913_LabelHand.tif |
| Sri-Lanka_85652 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Sri-Lanka_85652_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Sri-Lanka_85652_LabelHand.tif |
| USA_826217 | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/USA_826217_S1Hand.tif | https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/USA_826217_LabelHand.tif |

Pesos del modelo: facebook/sam-vit-base, https://huggingface.co/facebook/sam-vit-base (pesos oficiales de Meta, servidos via transformers de Hugging Face).

sha256 de los 20 archivos de chips descargados (para verificar integridad bit a bit):
```
042647898ddeb5be7fd46494c5bc413e22413f21465b90d0d13f57d16f23ffa0  chips/Ghana_5079_LabelHand.tif
5a598a0050e522fc8faf3ffbe8ce4edc3ad642d16c18ad1faa54574adf76f3c7  chips/Ghana_5079_S1Hand.tif
b0f21694aa0a1e5a48ccc4f66e12f5a8b3f16e606164e6cafd789b6aed121ba9  chips/India_1050276_LabelHand.tif
14a6d3889427e4b3e4a2b13fbc664cbc9bc84c33a3f2a63553b67cc0743bb73f  chips/India_1050276_S1Hand.tif
e90aaedbb51c617c6250c06cbeff13a14c992c46e5df1fe5b8988a5b239308b6  chips/Mekong_1149855_LabelHand.tif
e53925ce5265c89099b9d3aa1915661173ffd34375c751e2e24c1b1a4bd07587  chips/Mekong_1149855_S1Hand.tif
c8da655114d488f6097de39492da9e27f065feef58764c70ef5d7b8683ae8542  chips/Nigeria_31096_LabelHand.tif
928f806af7f3713ffce13c152f09713ab25198b646902ea0fe4e780a8ab70350  chips/Nigeria_31096_S1Hand.tif
60fe4edb4e6ce61fad7cd3e8810e0f030712fccf2053e7996c786941ffcb0130  chips/Pakistan_43105_LabelHand.tif
1594a4ab57da8f3ef16816661bdbba94669236364605465832047a8560c2252c  chips/Pakistan_43105_S1Hand.tif
fcb5f3b21ae74bd8275226d843d6254a982448b7669f43dbff6fea44647db24a  chips/Paraguay_305760_LabelHand.tif
ff60d990884d9ac12ab66c6604a4a9b33d7829e9a9a1d332142cdd048ab042f8  chips/Paraguay_305760_S1Hand.tif
9b7cb1c8a378c193b1be9405c6ea200c38c3fc80300d190184c0226383c0afcb  chips/Somalia_12849_LabelHand.tif
da7ae40306d1615d3bd11627cfdf7086b6f3a7335c709a69a146e725e25484ee  chips/Somalia_12849_S1Hand.tif
27fedadf6819805196eb1061a21e11ccb7ba26648a2424917b9de3174d45f2e0  chips/Spain_1199913_LabelHand.tif
343e99bb775af47d2fb690d04c1c71c56c751f179c13aa56c84992fd4e6b740f  chips/Spain_1199913_S1Hand.tif
101a849cbddaa60858d1ef42bab6c88d762c5b4aad31ddb0a4b8f0d9bfedb842  chips/Sri-Lanka_85652_LabelHand.tif
401dedb1de802b0793ada0e2f7288b2bdf2ccd1abff647be9be0b96948229210  chips/Sri-Lanka_85652_S1Hand.tif
f2a6c8a54d0fe0e6f36ee7f8930f11c8c1190f11873e97275d33e3d89a81ba6d  chips/USA_826217_LabelHand.tif
7d39bd4c52057b00a368342756883e32ef288fcc533fbe1775a9357a9df3b205  chips/USA_826217_S1Hand.tif
```
