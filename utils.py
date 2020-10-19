import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb
import gc

def read_train(filename = './data/sales_train_evaluation.csv'):
    # num_columns = ['d_{}'.format(i) for i in range(1,1914)]
    if 'sales.pkl' not in os.listdir('./data'):
        columns = pd.read_csv(filename, header=0, nrows=0).columns
        num_columns = [col for col in columns if col.startswith('d_')]
        dtypes = ['uint16'] * len(num_columns)
        dtypes = dict(zip(num_columns, dtypes))
        df = pd.read_csv(filename, dtype=dtypes)
        df['id'] = df['id'].str[:-len("evaluation")] + "validation"
        df.to_pickle('./data/sales.pkl')
    else:
        df = pd.read_pickle('./data/sales.pkl')
    return df

def read_calendar(filename = './data/calendar.csv'):
    if 'calendar.pkl' not in os.listdir('./data'):
        df = pd.read_csv(filename, parse_dates=['date'])
        df.to_pickle('./data/calendar.pkl')
    else:
        df = pd.read_pickle('./data/calendar.pkl')
    return df

def read_prices(filename = './data/sell_prices.csv'):
    if 'sell_prices.pkl' not in os.listdir('./data'):
        df = pd.read_csv(filename)
        df['wm_yr_wk'] = df['wm_yr_wk'].astype('uint16')
        df['sell_price'] = (100 * df['sell_price']).astype('uint16')
        df.sort_values(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)
        df.to_pickle('./data/sell_prices.pkl')
    else:
        df = pd.read_pickle('./data/sell_prices.pkl')
    return df

# columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
class cat_encoder():
    def __init__(self):
        self.mapping = {}

    def encode(self, df, column):
        if column not in self.mapping:
            self.mapping[column] = {}
            for i, value in enumerate(sorted(list(set(df[column])))):
                self.mapping[column][value] = i
        else:
            # find values in df[column] that are not in mapping
            new_values = []
            for value in list(set(df[column])):
                if value not in self.mapping[column]:
                    new_values.append(value)
            # add mapping for them
            if new_values != []:
                # find largest used number
                max_number = max(list(self.mapping[column].values()))
                # add mapping for new values
                for i, new_value in enumerate(sorted(new_values)):
                    self.mapping[column][new_value] = max_number + i + 1

        new_column = df[column].map(self.mapping[column]).astype('uint16')
        return new_column

    def decode(self, df, column):
        tmp_mapping = {}
        for key, value in self.mapping[column].items():
            tmp_mapping[value] = key
        new_column = df[column].map(tmp_mapping)
        return new_column

def load_encoder():
    with open('encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    return encoder

def save_encoder(encoder):
    with open('encoder.pickle', 'wb') as filename:
        pickle.dump(encoder, filename, protocol=pickle.HIGHEST_PROTOCOL)
    pass

def prepare_data(value_filter={}, mode='validation'):
    # read and filter train data
    sales = read_train()
    print(f'Rows before filter: {len(sales)}')
    if value_filter: # if dictionary is not empty
        for column, values in value_filter.items():
            if type(values) != list:
                values = [values]
            mask = sales[column].isin(values)
            sales = sales[mask]
    print(f'Rows after filter: {len(sales)}')

    # read and filter prices
    prices = read_prices()
    if value_filter: # if dictionary is not empty
        if 'store_id' in value_filter:
            column = 'store_id'
            values = value_filter[column]
            if type(values) != list:
                values = [values]
            mask = prices[column].isin(values)
            prices = prices[mask]

    calendar = read_calendar()

    # encode categorical feature to store less space

    encoder = load_encoder()
    # encoder = cat_encoder()
    for column in ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        sales[column] = encoder.encode(sales, column=column)
    for column in ['store_id', 'item_id']:
        prices[column] = encoder.encode(prices, column=column)

    # add forecast dates
    if mode == 'validation':
        fcst_date = 1914
    elif mode == 'evaluation':
        fcst_date = 1914 + 28
    for i in range(fcst_date, fcst_date + 28):
        sales[f'd_{i}'] = 0

    ### transform into time series data
    # save information about categorical variables into separate table
    summary = sales[['id', 'item_id', 'dept_id','cat_id','store_id','state_id']]
    # remove from the original table
    sales.drop(['item_id', 'dept_id','cat_id','store_id','state_id'],axis=1,inplace=True)
    # transform wide to long
    sales = sales.melt(id_vars='id', value_vars=sales.columns[1:], var_name='d', value_name='qty')
    sales['qty'] = sales['qty'].astype('uint16')
    # map days to dates
    d_to_date = dict(calendar[['date','d']].set_index('d')['date'])
    sales['d'] = sales['d'].map(d_to_date)
    # rename mapped column
    sales.rename(columns={'d':'date'}, inplace=True)
    # return categorial features
    sales = sales.join(summary.set_index('id'), on='id')

    return sales, prices, calendar, encoder

def day_to_week(day):
    return (day + 4) // 7

def create_abt(value_filter={}):
    sales, prices, calendar, encoder = prepare_data(value_filter)
    print('Data successfully prepared')

    # calculated date of the first sale and add it to summary and sales
    def add_feature_first_sale_date(df):
        firstbuy = df[df['qty']!=0].groupby('id')[['date']].min()
        firstbuy.rename(columns={'date':'firstbuy_date'}, inplace=True)
        dtmp = df[['id']].join(firstbuy, on='id', how='left')
        df['firstbuy_date'] = dtmp['firstbuy_date'].values
        pass

    add_feature_first_sale_date(sales)

    # remove starting zeroes for every TS
    sales = sales[sales['date'] >= sales['firstbuy_date']]

    # sort values by id and date to add lags
    sales.sort_values(['id', 'date'], inplace=True)

    # add day and week (number) features
    def add_features_day_week_num(sales, calendar):
        calendar['day'] = calendar['d'].apply(lambda x: int(x[2:])).astype('uint16')
        calendar['week'] = day_to_week(calendar['day']).astype('uint16')
        dtmp = sales[['date']].join(calendar[['date', 'day', 'week']].set_index('date'), on='date', how='left')
        sales['day'] = dtmp['day'].values
        sales['week'] = dtmp['week'].values
        pass
    
    add_features_day_week_num(sales, calendar)

    # add lag features
    def add_features_lags(sales):
        temp = sales[['id', 'day', 'qty']]
        temp.set_index(['id', 'day'], inplace=True)
        # сначала расчет фичей, потом джоины с шифтами
        sales_sum = pd.DataFrame({})
        # суммарные продажи за период (целые числа, можно хранить в uint16)
        for window in [7, 28, 91, 364]:
            temp2 = temp.groupby('id').apply(lambda x: x.rolling(window, min_periods=1).sum()).reset_index()
            temp2.rename(columns={'qty' : f'qty_sum_{window}'}, inplace=True)
            # reduce memory usage
            for column in temp2:
                temp2[column] = temp2[column].astype('uint16')
            # combine
            temp2.set_index(['id', 'day'], inplace=True)
            if sales_sum.empty:
                sales_sum = temp2
            else:
                sales_sum = sales_sum.join(temp2, on=['id', 'day'])

        # std
        for window in [28, 364]:
            temp2 = temp.groupby('id').apply(lambda x: x.rolling(window, min_periods=1).std()).reset_index()
            temp2.rename(columns={'qty' : f'qty_std_{window}'}, inplace=True)
            # reduce memory usage
            for column in temp2:
                temp2[column] = temp2[column].astype('float32')
            # combine
            temp2.set_index(['id', 'day'], inplace=True)
            sales_sum = sales_sum.join(temp2, on=['id', 'day'])

        # медианные продажи
        for window in [28, 364]:
            temp2 = temp.groupby('id').apply(lambda x: x.rolling(window, min_periods=1).median()).reset_index()
            temp2.loc[:, 'qty'] = temp2['qty'] * 2 # double to keep as int
            temp2.rename(columns={'qty' : f'qty_median_{window}'}, inplace=True)
            # reduce memory usage
            for column in temp2:
                temp2[column] = temp2[column].astype('uint16')
            # combine
            temp2.set_index(['id', 'day'], inplace=True)
            sales_sum = sales_sum.join(temp2, on=['id', 'day'])

        # отношения продаж за разные периоды
        for period in [28, 91, 364]:
            sales_sum[f'qty_sum_ratio_7_{period}'] = (sales_sum['qty_sum_7'] / sales_sum[f'qty_sum_{period}']).astype('float32')
            if period != 28:
                sales_sum[f'qty_sum_ratio_28_{period}'] = (sales_sum['qty_sum_28'] / sales_sum[f'qty_sum_{period}']).astype('float32')

        # джоин полученных фичей с различными шифтами (7, 14, 21, 28)
        periods = 0
        columns = [x for x in sales_sum.columns if x not in ['id', 'day']]
        for i in range(4):
            sales_sum.reset_index(inplace=True)
            sales_sum.loc[:, 'day'] = sales_sum['day'] + 7
            periods += 7
            sales_sum.set_index(['id', 'day'], inplace=True)
            sales_sum.columns = [x + f"_{periods}" for x in columns]
            sales_sum = sales[['id', 'day']].join(sales_sum, on=['id', 'day'])
            if 'id' in sales_sum.columns and 'day' in sales_sum.columns:
                sales_sum.set_index(['id', 'day'], inplace=True)
        #     sales.set_index(['id', 'day'], inplace=True)
            for column in sales_sum.columns:
                sales[column] = sales_sum[column].fillna(0).values
        #     sales.reset_index(inplace=True)

        for column in sales.columns:
            if 'sum' in column or 'median' in column:
                sales[column] = sales[column].astype('uint16')

        # лаги
        temp.reset_index(inplace=True)
        for lag in [7, 14, 21, 28, 35, 42, 49, 56, 364]:
            temp3 = temp.copy()
            temp3['day'] = temp3['day'] + lag
            column = f'lag_{lag}'
            temp3.rename(columns={'qty' : column}, inplace=True)
            temp3.set_index(['id', 'day'], inplace=True)
            temp3 = sales[['id', 'day']].join(temp3, on=['id', 'day'])
            sales[column] = temp3[column].fillna(0).astype('uint16')
        pass
        
    add_features_lags(sales)
    print('Lags successfully added')

    # events
    def add_features_events(sales, calendar):
        # divide events into separate columns by type
        for event_type in calendar['event_type_1'].dropna().unique():
            event_name_bytype = []
            for index, row in calendar.iterrows():
                if row['event_type_1'] == event_type:
                    event_name_bytype.append(row['event_name_1'])
                elif row['event_type_2'] == event_type:
                    event_name_bytype.append(row['event_name_2'])
                else:
                    event_name_bytype.append(np.nan)
            event_type = event_type.lower()
            calendar[f'event_name_{event_type}'] = event_name_bytype
        
        calendar.drop(['event_name_1',  'event_name_2', 'event_type_1', 'event_type_2'], axis=1, inplace=True)
        calendar.fillna(method='backfill', inplace=True)

        for column in ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']:
            tmp = calendar[['year', column, 'day']].groupby(['year', column]).max().rename(columns={'day':'maxday'})
            dtmp = calendar[['year', column]].join(tmp, on=['year', column])
            calendar['maxday'] = dtmp['maxday'].values
            calendar["days_till_next_{}".format(column.split('_')[2])] = calendar['maxday'] - calendar['day']
            calendar.drop('maxday', axis=1, inplace=True)

        dtmp = sales[['day']].join(calendar[['day','wday', 'month', 'year', 'wm_yr_wk',
                'event_name_cultural', 'event_name_sporting', 
                'event_name_national', 'event_name_religious',
                'days_till_next_cultural', 'days_till_next_sporting',
                'days_till_next_national', 'days_till_next_religious']].set_index('day'), on='day')
        for column in dtmp.columns:
            if column not in ['day']:
                sales[column] = dtmp[column].values
        for column in ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']:
            sales[column] = encoder.encode(sales, column)
        for column in ['days_till_next_cultural', 'days_till_next_sporting', 'days_till_next_national', 'days_till_next_religious']:
            sales[column] = sales[column].astype('uint16')
        for column in ['wday', 'month']:
            sales[column] = sales[column].astype('uint8')
        for column in ['wm_yr_wk', 'year']:
            sales[column] = sales[column].astype('uint16')
        pass
    
    add_features_events(sales, calendar)
    print('Events successfully added')

    # add length of history
    sales['history_length'] = ((sales['date'] - sales['firstbuy_date']).dt.days).astype('int16')
    
    sales.drop('firstbuy_date', axis=1, inplace=True)

    # add day number in year
    sales['dayofyear'] = (sales['date'].dt.dayofyear).astype('uint16')

    # combine 3 snap columns into 1
    def add_feature_snap(sales, calendar):
        snap_columns = [x for x in calendar.columns if x.startswith('snap_')]
        temp = calendar[['day'] + snap_columns]
        state_ids = list(encoder.mapping['state_id'].keys())
        dtmps = pd.DataFrame({})
        for state_id in state_ids:
            snap_current = [x for x in snap_columns if x.endswith(state_id)][0]
            dtmp = temp[['day', snap_current]]
            dtmp.loc[:, 'state_id'] = encoder.mapping['state_id'][state_id]
            dtmp['snap'] = dtmp[snap_current].values
            dtmp.drop(snap_current, axis=1, inplace=True)
            dtmps = pd.concat([dtmps, dtmp])
        
        columns = ['day', 'state_id']
        dtmps.set_index(columns, inplace=True)
        dtmp = sales[columns].join(dtmps, on=columns)
        sales['snap'] = dtmp['snap'].values
        sales['snap'] = sales['snap'].astype('uint8')
        pass

    add_feature_snap(sales, calendar)
    print('Snaps successfully added')

    def add_features_price(sales, prices):
        # group by store, item
        temp = prices.copy()
        temp.loc[:, 'wm_yr_wk'] = temp.loc[:, 'wm_yr_wk'] + 1
        temp.set_index(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)
        # calculate monthly mean
        temp = temp.groupby(['store_id', 'item_id']).apply(lambda x: x.rolling(4, min_periods=1).mean())
        temp.rename(columns={'sell_price':'price_mean_month'}, inplace=True)
        temp['price_mean_month'] = temp['price_mean_month'].astype(int)

        columns = ['store_id', 'item_id', 'wm_yr_wk']
        temp = temp.join(prices.set_index(columns), on=columns, how='left')
        # calculate ratio
        temp['price_mean_ratio_week_to_month'] = (temp['sell_price'] / temp['price_mean_month'])

        # price_columns = temp.columns
        # for period in [7, 14, 21, 28]:
        #     # adjust week depending on horizon
        #     temp.reset_index(inplace=True)
        #     temp['wm_yr_wk'] = temp['wm_yr_wk'] + 1
        #     temp.set_index(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)
        #     temp.columns = [x + f'_{period}' for x in price_columns]

        #     dtmp = sales[columns].join(temp, on=columns)
        #     dtmp.set_index(columns, inplace=True)
        #     dtmp = dtmp.groupby(['store_id', 'item_id']).ffill().groupby(['store_id', 'item_id']).bfill()
        #     for column in dtmp.columns:
        #         if column not in columns:
        #             sales[column] = dtmp[column].values
        #             if 'sell_price' in column:
        #                 sales[column] = sales[column].astype('uint16')
        #             elif 'price_mean_month' in column:
        #                 sales[column] = sales[column].astype('uint16')
        dtmp = sales[columns].join(temp, on=columns)
        dtmp.set_index(columns, inplace=True)
        dtmp = dtmp.groupby(['store_id', 'item_id']).ffill().groupby(['store_id', 'item_id']).bfill()
        for column in dtmp.columns:
            if column not in columns:
                sales[column] = dtmp[column].values
                if 'sell_price' in column:
                    sales[column] = sales[column].astype('uint16')
                elif 'price_mean_month' in column:
                    sales[column] = sales[column].astype('uint16')
                elif 'price_mean_ratio' in column:
                    sales[column] = sales[column].astype('float32')
        pass
    
    add_features_price(sales, prices)
    print('Price features successfully added')

    # sales.dropna(inplace=True)
    sales.drop(['id', 'date'], axis=1, inplace=True)
    sales.reset_index(drop=True, inplace=True)

    sales['item_id'] = sales['item_id'].astype('uint16')
    sales['store_id'] = sales['store_id'].astype('uint8')

    print("don't forget to fill missing holidays: matters at evaluation")
    print("don't forget to check which rows get removed")

    save_encoder(encoder)
    return sales




# masterfile adventures
def read_master(masterfile = 'master.csv'):
    df = pd.read_csv(masterfile, dtype={'model_id':str,'parameters':str})
    return df

def save_master(df, masterfile = 'master.csv'):
    df.to_csv(masterfile, index=False)
    print('Masterfile was updated')

def add_model(parameters, masterfile = 'master.csv'):
    '''
    Adds model to master file and gives it unique id
    '''
    if masterfile in os.listdir():
        df = read_master()
        if str(parameters) in list(df['parameters']):
            print(f'Model already included. No changes were made to {masterfile}')
            return False
        else:
            max_index = df['model_id'].astype(int).max()
            new_id = max_index + 1
            df.loc[new_id, 'model_id'] = '0'*(4-len(str(new_id))) + str(new_id)
            df.loc[new_id, 'parameters'] = str(parameters)
    else:
        print(f'Masterfile was not found and will be created at ./{masterfile}')
        df = pd.DataFrame({'model_id' : '0000', 'parameters' : str(parameters)}, index=[0])
    save_master(df)
    return True

def get_model_id(parameters, masterfile = 'master.csv'):
    '''
    Finds model_id with specified parameters in master file 
    '''
    df = read_master()
    return df.loc[df['parameters'] == str(parameters), 'model_id'].values[0]

def get_model_parameters(model_id, masterfile = 'master.csv'):
    '''
    Returns parameters of the model with specified model_id from the masterfile
    '''
    df = read_master()
    return eval(df.loc[df['model_id'] == str(model_id), 'parameters'].values[0])

def check_model(model_id, months=[1,2,3], masterfile = 'master.csv'):
    '''
    Checks whether model is evaluated: prediction file exists
    '''
    flag = 0
    for month in months:
        name = f'{model_id}_{month}.csv'
        if name in os.listdir('./forecasts/'):
            print(f'{model_id} is evaluated for month {month}')
        else:
            flag += 1
            print(f'{model_id} is not evaluated for month {month}')
    if flag == 0:
        return True
    else:
        return False

def evaluate_model(model_id, months=[1, 2, 3]):
    '''
    Trains lgbm model given parameters and outputs model and prediction
    '''
    parameters = get_model_parameters(model_id)
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    cat_feats = cat_feats + ['wday', 'month']
    cat_feats = cat_feats + ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']

    # state_ids = ['CA', 'TX', 'WI']
    store_ids = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

    if 'by_variable' in list(parameters.keys()):
        by_variable = parameters['by_variable']
    else:
        by_variable = 'state_id'
    encoder = load_encoder()

    print(f'Models are divided by {by_variable}')
    values = list(encoder.mapping[by_variable].keys())
    print(values)

    fcst_horizon = parameters['fcst_horizon']
    history_length = parameters['history_length']
    history_start = parameters['history_start']
    i = 1
    for month in months:
        fcst_date = 1914 - 28 * month
        predictions = []
        for value in values:
            print(f'Step {i} out of {len(values) * len(months)}')
            # 1. read abt
            partial_abts = []
            for store_id in store_ids:
                partial_abt = pd.read_pickle(f'./work/abt_{store_id}.pkl')
                features = get_features(partial_abt, fcst_horizon)
                partial_abt = partial_abt[features]
                # apply by_variable filter
                partial_abt = partial_abt[partial_abt[by_variable] == encoder.mapping[by_variable][value]]
                # filter dates
                partial_abt = partial_abt[partial_abt['day'] >= fcst_date - history_length]
                partial_abt = partial_abt[partial_abt['day'] < fcst_date + 28]
                # filter start of every TS
                partial_abt = partial_abt[partial_abt['history_length'] >= history_start]

                partial_abts.append(partial_abt)
            abt = pd.concat(partial_abts)
            print('abt is successfully gathered')
            partial_abts = None
            partial_abt = None
            gc.collect()
            # if fcst_horizon == 4:
            #     with open(f'./work/abt_{fcst_horizon}_{state_id}.pkl', 'rb') as handle:
            #         abt = pickle.load(handle)
            # else:
            #     raise ValueError('Forecast horizons other than 4 not implemented yet')
            if 'Unnamed: 0' in abt.columns:
                abt.drop('Unnamed: 0', axis=1, inplace=True)
            # 2. filter according to state_id
            # abt = abt[encoder.decode(abt, 'state_id') == state_id]

            for feature in cat_feats:
                abt[feature] = abt[feature].astype('category')
            
            # 3.  separate into train and test

            if 'use_scaling' in parameters.keys() and parameters['use_scaling'] == 1:
                print("Scaling is used because parameter 'use_scaling' was set to 1")
                # get average sales for every TS over last year
                qty_scaling = abt.loc[abt['day'] == fcst_date + fcst_horizon * 7 - 6, ['item_id', 'store_id', 'qty_mean_lastyear']]
                qty_scaling.rename(columns={'qty_mean_lastyear':'qty_scaling'}, inplace=True)
                qty_scaling.set_index(['item_id', 'store_id'], inplace=True)
                qty_scaling.loc[qty_scaling['qty_scaling'] < 1/364, 'qty_scaling'] = 1
                qty_scaling = abt[['item_id','store_id']].join(qty_scaling, on=['item_id', 'store_id'])
                abt['qty_scaling'] = qty_scaling['qty_scaling']
                qty_scaling = None
                gc.collect()
                for feature in ['qty_mean_lastweek', 'qty_mean_lastmonth', 'qty_mean_lastyear', 'qty_std_lastyear', 'qty']:
                    abt[feature] = abt[feature] / abt['qty_scaling']
            else:
                print('Scaling is not used')

            y_train = abt.loc[abt['day'] < fcst_date, 'qty']
            X = abt.drop(['qty'], axis=1)
            X_train = X[X['day'] < fcst_date]
            X_test = X[X['day'] >= fcst_date]

            # 4. convert to lgbm dataset
            if 'use_weights' in parameters.keys() and parameters['use_weights'] == 1:
                weights = get_weights_for_abt(X_train, fcst_date)
                print('Weighting is used')
            else:
                print('Weighting is not used')

            print(f'Train data shape: {X_train.shape}')
            if 'use_weights' in parameters.keys() and parameters['use_weights'] == 1:
                train_data = lgb.Dataset(X_train, label = y_train, weight = weights, categorical_feature=cat_feats, free_raw_data=True)
            else:
                train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_feats, free_raw_data=True)
            # 4. train
            model = lgb.train(parameters['lgbm_params'], train_data)
            # 5. make prediction
            X_test['predict'] = model.predict(X_test)
            if 'use_scaling' in parameters.keys() and parameters['use_scaling'] == 1:
                 X_test['predict'] = X_test['predict'] * X_test['qty_scaling']
            cat_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            columns = cat_columns + ['day', 'predict']
            prediction = X_test[columns]
            for column in cat_columns:
                prediction.loc[:, column] = encoder.decode(prediction, column)

            # 6. save model and prediction
            name = f'{model_id}_{month}_{value}'
            model.save_model("./work/" + name + ".lgb")
            predictions.append(prediction)
            i += 1
        
        predictions = pd.concat(predictions)
        name = f'{model_id}_{month}'
        predictions.to_csv('./forecasts/pred_' + name + '.csv', index=False)
    print('Model evaluation complete')
    pass

def get_val_fcst(model_id):
    evaluate_model(model_id, months=[0])
    pass

def prepare_val_submission(model_id):
    df = pd.read_csv(f'./forecasts/pred_{model_id}_0.csv')
    df['id'] = df['item_id'] + '_' + df['store_id'] + '_validation'
    df = df[['id', 'day','predict']]
    df['day'] = df['day'] - 1913
    df['day'] = 'F' + df['day'].astype(str)
    df = df.pivot(index='id', columns='day', values='predict')
    df =df[[f'F{x}' for x in range(1,29)]]
    example = pd.read_csv('./data/sample_submission.csv')
    submission = example[['id']].join(df, on='id').fillna(0)
    submission.to_csv(f'./submissions/{model_id}.csv', index=False)
    pass

def prepare_val_weights():
    weights = pd.read_csv('./data/weights_validation.csv')
    weights = weights[weights['Level_id']=='Level12'].rename(columns={'Agg_Level_1':'item_id', 'Agg_Level_2':'store_id','Weight':'weight'})[['item_id','store_id','weight']]
    weights.to_csv('./work/validation_weights.csv', index=False)

def get_weights_for_abt(abt, fcst_date):
    history = load_history(fcst_date)
    denominator = get_rmsse_denominator(history, fcst_date)
    history['denominator'] = denominator
    history = history[['item_id','store_id','denominator']]
    history.set_index(['item_id','store_id'], inplace=True)

    weights = pd.read_csv('./work/validation_weights.csv')
    weights = weights.join(history, on=['item_id','store_id'])

    mask = (weights['weight'] != 0)
    weights['weight2'] = 0
    weights.loc[mask, 'weight2'] = weights.loc[mask, 'weight'] / np.sqrt(weights.loc[mask, 'denominator'])

    encoder = load_encoder()
    # encode to join with abt
    for column in ['store_id','item_id']:
        weights[column] = encoder.encode(weights, column)
    # add weight to abt
    weights.set_index(['store_id','item_id'], inplace=True)
    weights = weights[['weight2']]
    weights = weights * len(weights) / weights.sum() # rescale weights
    weights = abt[['store_id','item_id']].join(weights,on=['store_id','item_id'])
    return np.array(weights['weight2'])

# error calculation
def get_rmsse_denominator(history, fcst_date):
    # denominator in error calculation
    # calculate squared differences fron n=1 to fcst_date
    tmp = np.array(history[range(1, fcst_date)])
    tmp = tmp[:, 1:] - tmp[:, : -1]
    tmp = tmp ** 2
    # create mask matrix
    mask = np.array(history[range(1, fcst_date)])
    mask = np.cumsum(mask, axis=1)
    mask = (mask > 0)
    # multiply by mask = {0,1}
    tmp = tmp * mask[:, 1:]
    # sum
    tmp = tmp.sum(axis=1)
    # divide by coefficient
    first_sale = (mask == 0).sum(axis=1)
    tmp = tmp / (fcst_date - 1 - 1 -  first_sale)
    return tmp

def load_history(fcst_date):
    history  = read_train()
    cols_cat = [x for x in history.columns if not x.startswith('d_')]
    cols_qty = [int(x[2:]) for x in history.columns if x.startswith('d_')]
    history.columns = cols_cat + cols_qty
    cols_qty = [x for x in cols_qty if x < fcst_date + 28]
    history = history[cols_cat + cols_qty]
    history.set_index('id', inplace=True)
    return history

def get_features(abt, fcst_horizon=4):
    columns_all = ['qty', 'item_id','dept_id','cat_id','store_id','state_id','day','week','year','wm_yr_wk','month','wday']
    columns_all += ['history_length', 'dayofyear','snap']
    columns_all += ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']
    columns_all += ['days_till_next_cultural', 'days_till_next_sporting', 'days_till_next_national', 'days_till_next_religious']
    columns_all +=  ['price_mean_month', 'sell_price', 'price_mean_ratio_week_to_month']

    # лаги все что можно
    columns_lag = [x for x in abt.columns if x.startswith('lag') and int(x.split('_')[-1]) >= fcst_horizon * 7]

    # qty_sum_7 все что можно
    columns_qty = [x for x in abt.columns if x.startswith('qty') and x !='qty']
    columns_qty = [x for x in columns_qty if (int(x.split('_')[-1]) == fcst_horizon * 7) or (int(x.split('_')[-1]) > fcst_horizon * 7 and x.startswith('qty_sum_7'))]

    # abt.drop(columns_all, axis=1).info()
    columns_keep = columns_all + columns_lag + columns_qty
    return columns_keep

def load_summary():
    summary = pd.read_pickle('./work/summary.pkl')
    return summary

if __name__ == "__main__":
    pass
