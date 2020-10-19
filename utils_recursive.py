from utils import *
from tqdm import tqdm

def load_prices_pivot(fcst_date=None, history_length=730, value_filter={}):
    start_day = fcst_date - history_length - 364
    ### prepare calendar data to join with prices
    cal = read_calendar()
    cal['day'] = cal['d'].apply(lambda x: int(x[2:])).astype('uint16')
    cal.drop('d', axis=1,inplace=True)
    # filter
    cal = cal[['wm_yr_wk', 'day']].set_index('day')
    cal = cal.loc[start_day:, :]
    cal.reset_index(inplace=True)
    cal.set_index('wm_yr_wk', inplace=True)
    
    ### load summary to apply filter
    encoder = load_encoder()
    summary = load_summary()
    if value_filter:
        for key, values in value_filter.items():
            if type(values) != list:
                values = [values]
            summary = summary[summary[key].isin(values)]
    ids = encoder.encode(summary, 'id') # list of filtered ids
    summary = None
    
    ### prices
    prices = read_prices()
    # add id, remove item and store
    prices['id'] = prices['item_id'] + '_' + prices['store_id'] + '_validation'
    prices.drop(['item_id', 'store_id'],axis=1,inplace=True)
    # encode and filter id
    prices['id'] = encoder.encode(prices, 'id')
    prices = prices[prices['id'].isin(ids)]
    # join with cal
    prices = prices.join(cal, on='wm_yr_wk')
    prices.dropna(subset=['day'], inplace=True)
    # change 'day' and 'sell_price' to int, remove week number
    prices['sell_price'] = prices['sell_price'].astype('uint16')
    prices['day'] = prices['day'].astype('uint16')
    prices.drop('wm_yr_wk', axis=1, inplace=True)
    # transform table from long to wide
    prices = prices.pivot(index='id', columns='day',values='sell_price')
    # deal with missing prices
    prices = prices.ffill(axis=1).bfill(axis=1)
    prices = prices.loc[:, : fcst_date + 27]
    return prices


def create_price_features(prices, fcst_date=1914, history_length=730):
    ### создание фичей
    # 1. средняя цена за неделю (шифт 1)
    price_mean_7   = prices.rolling(7, axis=1).mean().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 2. средняя цена за месяц (шифт 1)
    price_mean_28 = prices.rolling(28, axis=1).mean().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 3. относительная волатильность цены за год
    price_rstd_364 = (prices.rolling(364, axis=1).std().shift(1, axis=1) / prices.rolling(364, axis=1).mean().shift(1, axis=1)).loc[:, fcst_date - history_length :]
    
    ### сбор фичей
    prices = prices.loc[:, fcst_date - history_length :]
    columns = list(prices.columns)

    prices.reset_index(inplace=True)
    prices = pd.melt(prices, id_vars=['id'], value_vars=columns, value_name='sell_price')

    dfs = [price_mean_7, price_mean_28, price_rstd_364]
    cols = ['price_mean_7', 'price_mean_28', 'price_rstd_364']
    for df, column in zip(dfs, cols):
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=['id'], value_vars=columns, value_name=column)
        df.set_index(['id', 'day'], inplace=True)
        df = prices[['id', 'day']].join(df, on= ['id', 'day'])
        prices[column] = df[column].values
        prices[column] = prices[column].astype('float32')
        df = None
    
    for column in ['id' ,'day','sell_price']:
        prices[column] = prices[column].astype('uint16')
    
    prices['price_ratio_sellprice_to_mean7'] = (prices['sell_price'] / prices['price_mean_7']).astype('float32')
    prices['price_ratio_sellprice_to_mean28'] = (prices['sell_price'] / prices['price_mean_28']).astype('float32')
    prices['price_ratio_mean7_to_mean28'] = (prices['price_mean_7'] / prices['price_mean_28']).astype('float32')
    return prices


def create_features_in_batches(df, function, fcst_date=None, history_length=730, n_batches=20):
    batchsize = int((df.shape[0] + 1)/ n_batches) + 1
    features = []
    for i in tqdm(range(n_batches)):
        batch = df.iloc[batchsize * i : batchsize * (i + 1)]
        features.append(function(batch, fcst_date=fcst_date, history_length=history_length))
    features = pd.concat(features)
    features.reset_index(drop=True, inplace=True)
    return features


def load_sales_pivot(fcst_date=None, history_length=730, value_filter={}):
    # load data
    sales = read_train()
    # apply value_filter
    if value_filter:
        for key, values in value_filter.items():
            if type(values) != list:
                values = [values]
            sales = sales[sales[key].isin(values)]
    # drop categorical columns
    sales.drop(['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id'],axis=1,inplace=True)
    # encode id
    encoder = load_encoder()
    sales['id'] = encoder.encode(sales, 'id')
    sales.set_index('id', inplace=True)
    # add fcst days
    for day in range(fcst_date, fcst_date + 28):
        sales[f'd_{day}'] = 0
        sales[f'd_{day}'] = sales[f'd_{day}'].astype('uint16')
    # convert days from str to int and filter
    sales.columns = [int(x[2:]) for x in sales.columns]
    start_day = fcst_date - history_length - 364
    sales = sales.loc[:, start_day:]
    sales.sort_index(inplace=True)
    return sales

def create_sales_features(sales, fcst_date=None, history_length=730):
    ### создание фичей
    # 0. day of the first sale
    first_sale = ((sales.loc[:, 1:].copy().cumsum(axis=1) == 0).sum(axis=1) + 1).astype('uint16')
    first_sale = pd.DataFrame(first_sale).reset_index().rename(columns={'index':'id', 0:'first_sale_day'}).set_index('id')
    # 1. средние продажи за неделю (шифт 1)
    sales_mean_7   = sales.rolling(7, axis=1).mean().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 2. средние продажи за месяц (шифт 1)
    sales_mean_28 = sales.rolling(28, axis=1).mean().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 3. средние продажи за год (шифт 1)
    sales_mean_364 = sales.rolling(364, axis=1).mean().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 4. относительная волатильность продаж за месяц
    sales_rstd_28 = sales.rolling(28, axis=1).std().shift(1, axis=1).loc[:, fcst_date - history_length :]
    sales_rstd_28 = sales_rstd_28 / sales_mean_28
    sales_rstd_28[sales_rstd_28 == np.inf] = 0
    # 5. относительная волатильность продаж за год
    sales_rstd_364 = sales.rolling(364, axis=1).std().shift(1, axis=1).loc[:, fcst_date - history_length :]
    sales_rstd_364 = sales_rstd_364 / sales_mean_364
    sales_rstd_364[sales_rstd_364 == np.inf] = 0
    # 6. медианные продажи за месяц
    sales_median_28 = sales.rolling(28, axis=1).median().shift(1, axis=1).loc[:, fcst_date - history_length :]
    # 7. медианные продажи за год
    sales_median_364 = sales.rolling(364, axis=1).median().shift(1, axis=1).loc[:, fcst_date - history_length :]
    
    # lags
    lags = [1, 6, 7, 13, 14, 20, 21, 28, 42, 49, 64, 65, 364, 365]
    cols_lags = ['lag' + str(x) for x in lags]
    lag_dfs = []
    for lag in lags:
        lag_dfs.append(sales.shift(lag, axis=1).loc[:, fcst_date - history_length :])

    ### сбор фичей
    sales = sales.loc[:, fcst_date - history_length :]
    columns = list(sales.columns)

    sales.reset_index(inplace=True)
    sales = pd.melt(sales, id_vars=['id'], value_vars=columns, var_name='day', value_name='sales')
    
    # add first sale day
    sales = sales.join(first_sale, on='id')
    
    # add sales features
    dfs = [sales_mean_7, sales_mean_28, sales_mean_364, sales_rstd_28, sales_rstd_364, sales_median_28, sales_median_364]
    dfs = dfs + lag_dfs
    cols = ['sales_mean_7', 'sales_mean_28', 'sales_mean_364', 'sales_rstd_28', 'sales_rstd_364', 'sales_median_28', 'sales_median_364']
    cols = cols + cols_lags
    for df, column in zip(dfs, cols):
        df.reset_index(inplace=True)
        df = pd.melt(df, id_vars=['id'], value_vars=columns, var_name='day',value_name=column)
        df.set_index(['id', 'day'], inplace=True)
        df = sales[['id', 'day']].join(df, on= ['id', 'day'])
        sales[column] = df[column].values
        if column.startswith('lag'):
            # sales[column] = sales[column].astype('uint16')
            sales[column] = sales[column].astype('float32')
        else:
            sales[column] = sales[column].astype('float32')
        df = None
    
    for column in ['id' ,'day','sales']:
        sales[column] = sales[column].astype('uint16')
    
    sales['sales_ratio_mean7_to_mean28'] = (sales['sales_mean_7'] / sales['sales_mean_28']).astype('float32')
    sales['sales_ratio_mean7_to_mean364'] = (sales['sales_mean_7'] / sales['sales_mean_364']).astype('float32')
    sales['sales_ratio_mean28_to_mean364'] = (sales['sales_mean_28'] / sales['sales_mean_364']).astype('float32')
    
    sales.fillna(0, inplace=True)
    # add history length
    sales['history_length'] = (sales['day'] - sales['first_sale_day']).astype('uint16')
    return sales


def create_features_events():
    calendar = read_calendar()
    calendar['dayofyear'] = (calendar['date'].dt.dayofyear).astype('uint16')
    calendar['day'] = calendar['d'].apply(lambda x: int(x[2:]))
    
    # divide events into separate columns by type
    for event_type in calendar['event_type_1'].dropna().unique():
        event_name_bytype = []
        for _, row in calendar.iterrows():
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

    calendar['event_name_religious'].fillna('Eid al-Fitr', inplace=True)
    feature = 'days_till_next_religious'
    days = calendar[calendar[feature].isna()].index
    calendar.loc[days, feature] = calendar.loc[days, 'date'].apply(lambda x: (pd.to_datetime('2016-07-07') - x).days)
    
    for feature in ['event_name_national', 'days_till_next_national']:
        days = calendar.loc[calendar[feature].isna(), 'day'].values
        days = [x - 365 for x in days]
        calendar.loc[calendar[feature].isna(), feature] = calendar.loc[calendar['day'].isin(days), feature].values
    
    encoder = load_encoder()
    
    for column in ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']:
        calendar[column] = encoder.encode(calendar, column)
        calendar[column] = calendar[column].astype('uint8')
    for column in ['days_till_next_cultural', 'days_till_next_sporting', 'days_till_next_national', 'days_till_next_religious']:
        calendar[column] = calendar[column].astype('uint16')
    for column in ['wday', 'month', 'snap_CA', 'snap_TX', 'snap_WI']:
        calendar[column] = calendar[column].astype('uint8')
    for column in ['year', 'day']:
        calendar[column] = calendar[column].astype('uint16')
    
    calendar.drop(['date', 'd', 'wm_yr_wk', 'weekday'], axis=1, inplace=True)
    return calendar


def create_abt_recursive(fcst_date=None, value_filter={'cat_id' : 'FOODS'}, history_length=730, n_batches=20):
    print('Preparing price features...')
    prices = load_prices_pivot(fcst_date, history_length, value_filter)
    price_features = create_features_in_batches(prices, create_price_features, fcst_date, history_length, n_batches)

    print('Preparing sales features...')
    sales = load_sales_pivot(fcst_date=fcst_date, value_filter=value_filter)
    sales_features = create_features_in_batches(sales, create_sales_features, fcst_date, history_length, n_batches)

    print('Preparing event features...')
    event_features = create_features_events()
    print('Done')
    
    print('Preparing categorical features...')
    summary = load_summary()
    encoder = load_encoder()
    for col in summary.columns:
        summary[col] = encoder.encode(summary, col)
    print('Done')

    print('Combining features into abt ...')
    price_features = price_features.join(sales_features.set_index(['id', 'day']), on = ['id', 'day'])
    price_features = price_features.join(event_features.set_index('day'), on='day')
    price_features = price_features.join(summary.set_index('id'), on='id')
    # dropping useless features
    # price_features.drop(['id'], axis=1, inplace=True)
    print('Done')
    return price_features


def train_model_recursive(model_id, months=[0]):
    '''
    Trains lgbm model given parameters and outputs model and prediction
    '''
    parameters = get_model_parameters(model_id)
    if parameters['recursive'] != True:
        return False
    cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    cat_feats = cat_feats + ['wday', 'month']
    cat_feats = cat_feats + ['event_name_cultural', 'event_name_sporting', 'event_name_national', 'event_name_religious']
    cat_feats = [x for x in cat_feats if x not in parameters['remove_features']]


    if 'by_variable' in list(parameters.keys()):
        by_variable = parameters['by_variable']
    else:
        by_variable = 'state_id'
    encoder = load_encoder()

    print(f'Models are divided by {by_variable}')
    values = list(encoder.mapping[by_variable].keys())
    print(values)

    history_length = parameters['history_length']
    history_start = parameters['history_start']
    i = 1
    for month in months:
        fcst_date = 1914 - 28 * month
        for value in values:
            print(f'Step {i} out of {len(values) * len(months)}')
            print(by_variable, ':', value)
            value_filter = {by_variable : value}
            
            ### load abt
            abtname = f'abt_recursive_{fcst_date}_{history_length}_{value}.pkl'
            if abtname not in os.listdir('./work'):
                # if abt is not in work - create new and save
                abt = create_abt_recursive(value_filter=value_filter, fcst_date=fcst_date, history_length=history_length, n_batches=100)
                abt.to_pickle(f'./work/{abtname}')
            else:
                # otherwise - load pickle
                abt = pd.read_pickle(f'./work/{abtname}')
            
            # abt.drop('id', axis=1, inplace=True)
            abt.drop(parameters['remove_features'], axis=1, inplace=True)
            abt = abt[abt['history_length'] >= history_start]
            
#             for feature in cat_feats:
#                 abt[feature] = abt[feature].astype('category')
            
            # 3.  separate into train and test
            y_train = abt.loc[abt['day'] < fcst_date, 'sales']
            X = abt.drop(['sales'], axis=1)
            X_train = X[X['day'] < fcst_date]
            # X_test = X[X['day'] >= fcst_date]

            # 4. convert to lgbm dataset
            print(f'Train data shape: {X_train.shape}')
            train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_feats, free_raw_data=True)
            val_data = lgb.Dataset(X_train.iloc[:1000], label = y_train[:1000], categorical_feature=cat_feats, free_raw_data=True)
            X_train = None
            # 5. train
            model = lgb.train(parameters['lgbm_params'], train_data, valid_sets = [val_data], verbose_eval=20)

            # 6. save model
            name = f'{model_id}_{month}_{value}'
            model.save_model("./work/" + name + ".lgb")
            i += 1
    print('Model training complete')
    pass


def update_features(abt, fcst_day):
    idx = (abt['day'] == fcst_day)
    ### columns to update
    ## lags
    for lag in [int(x[3:]) for x in abt.columns if x.startswith('lag')]:
        # new _values
        lag_col = abt.loc[abt['day'] == (fcst_day - lag), 'sales'].values
        # old values
        abt.loc[idx, f'lag{lag}'] = lag_col

    ## sales mean
    # for period in [7, 28, 364]:
    #     # update mean as += lag1 / period
    #     # old values
    #     old_values = abt.loc[idx, f'sales_mean_{period}'].values
    #     delta_values = abt.loc[idx, 'lag1'].values / period
    #     new_values = old_values + delta_values
    #     # update
    #     abt.loc[abt['day'] == fcst_day, f'sales_mean_{period}'] = new_values

    # sales median and rstd and mean
    for period in [7, 28, 364]:
        day_range = range(fcst_day - period, fcst_day)
        # take last {period} values
        tmp = abt.loc[abt['day'].isin(day_range), ['sales', 'id', 'day']]
        # convert to pivot
        tmp = tmp.pivot(index='id', columns='day',values='sales')

        mean = tmp.mean(axis=1).values
        abt.loc[idx, f'sales_mean_{period}'] = mean
        if period != 7:
            # recalculate median and rstd
            median = tmp.median(axis=1).values
            rstd = tmp.std(axis=1).values / tmp.mean(axis=1).values
            # update
            abt.loc[idx, f'sales_median_{period}'] = median
            abt.loc[idx, f'sales_rstd_{period}'] = rstd

    # sales ratio
    abt.loc[idx, 'sales_ratio_mean7_to_mean28'] = abt.loc[idx, 'sales_mean_7'] / abt.loc[idx, 'sales_mean_28']
    abt.loc[idx, 'sales_ratio_mean7_to_mean364'] = abt.loc[idx, 'sales_mean_7'] / abt.loc[idx, 'sales_mean_364']
    abt.loc[idx, 'sales_ratio_mean28_to_mean364'] = abt.loc[idx, 'sales_mean_28'] / abt.loc[idx, 'sales_mean_364']
    return abt


def get_prediction_recursive(abt, m, model_id, fcst_date = 1914):
    abt = abt[abt['day'] >= fcst_date - 364]
    parameters = get_model_parameters(model_id)
    prediction = pd.DataFrame({})
    for fcst_day in tqdm(range(fcst_date, fcst_date + 28)):
        idx = (abt['day'] == fcst_day)
        # get forecast for next day
        dX = abt.loc[idx, :].drop(['sales'] + parameters['remove_features'], axis=1)
        pred = m.predict(dX)
        # write false history back to abt
        abt.loc[idx, 'sales'] = pred
        # save prediction
        d_pred = abt.loc[idx, ['id', 'sales']].rename(columns={'sales' : fcst_day}).set_index('id')
        if prediction.empty:
            prediction = d_pred
        else:
            prediction = prediction.join(d_pred, on='id')
        # update features
        if fcst_day != fcst_date + 28 - 1:
            abt = update_features(abt, fcst_day = fcst_day + 1)
    return prediction



def predict_and_prepare_submit_recursive(model_id, month):
    ### get prediction
    parameters = get_model_parameters(model_id)
    history_length = parameters['history_length']
    history_start = parameters['history_start']
    fcst_date = 1914 - 28 * month

    if 'by_variable' in list(parameters.keys()):
        by_variable = parameters['by_variable']
    else:
        by_variable = 'state_id'
    encoder = load_encoder()
    values = list(encoder.mapping[by_variable].keys())

    prediction = []
    for value in values:
        print(f'Preparing prediction for {value}')
        abt = pd.read_pickle(f'./work/abt_recursive_{fcst_date}_{history_length}_{value}.pkl')
        abt.sort_values(['id', 'day'], inplace=True)
        m = lgb.Booster(model_file=f'./work/{model_id}_{month}_{value}.lgb')
        prediction.append(get_prediction_recursive(abt, m, model_id, fcst_date = fcst_date))

    prediction = pd.concat(prediction)
    prediction.to_csv("./fin_submission/final_prediction.csv")
    ### submit prediction
    prediction.columns = [f"F{x+1-fcst_date}" for x in prediction.columns]
    prediction.reset_index(inplace=True)
    prediction['id'] = encoder.decode(prediction, 'id')
    prediction.set_index('id', inplace=True)

    example = pd.read_csv('./data/sample_submission.csv')
    submission = example[['id']].join(prediction, on='id')
    prediction.reset_index(inplace=True)
    prediction['id'] = prediction['id'].str[:-len("validation")] + "evaluation"
    # prediction.set_index('id', inplace=True)
    submission = pd.concat([submission, prediction])
    submission.to_csv(f'./fin_submission/{model_id}.csv', index=False)
    pass
