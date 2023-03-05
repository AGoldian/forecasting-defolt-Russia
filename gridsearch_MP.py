import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import *
from darts.metrics import *
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.catboost_model import CatBoostModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

df = pd.read_csv(r'all_features_with_swan.csv')
df.drop(columns=['is_swan_event_covid', 'is_swan_event_default', 'is_swan_event_oil', 'is_swan_event_crimea'], inplace=True)

def covariates(data: pd.DataFrame) -> list:

    true_columns = list(data.columns)
    true_columns.remove('Отчетная дата')
    
    train_data = []
    for column in true_columns:
        ts = TimeSeries.from_dataframe(data[['Отчетная дата', column]], time_col='Отчетная дата', fillna_value=True)
        scaler = Scaler()

        data[column] = np.array(data[column], dtype=np.int64)
        train_data.append(scaler.fit_transform(ts))
        
        if column == 'Всего':
            global main_scaler
            main_scaler = scaler

    
    return train_data

global_best_r2s = -1000
global_best_params = None


def gridsearch_MP(lag):

    dataframe = df
    local_best_r2s = -1000
    better_param = None

    for chunk in tqdm(range(1, 13 + 1)):
            tmp_model = CatBoostModel(lags=lag,
                                      output_chunk_length=chunk)
            
            main_data = covariates(dataframe)
            train_data, val_data = main_data[0][:-24], main_data[0][-24:]

            tmp_model.fit(series=main_data, verbose=False)

            pred_data = tmp_model.predict(series=train_data, n=24)
            pred_data = main_scaler.inverse_transform(pred_data)
            val_data = main_scaler.inverse_transform(val_data)

            r2s = r2_score(val_data, pred_data)

            if r2s > local_best_r2s:
                local_best_r2s = r2s
                better_param = (chunk, lag)


            graphic_pred = tmp_model.predict(series=main_data[0], n=96)
            graphic_pred.plot(label='Прогноз')
            plt.savefig(f'graphics/{chunk}_{len(lag)}_{lag}.png')

    
            print(f'Calculate model #{chunk} done!')
    return [local_best_r2s, better_param]




if __name__ == '__main__':

    lag_list = [[-i for i in range(1, j)] for j in range(2, 64 + 1)]
    with Pool() as pool:
        best_score = [object for object in pool.imap_unordered(gridsearch_MP, lag_list)]

    global_best_r2s = [data for data in best_score]
    print(global_best_r2s)
    print(max(global_best_r2s))