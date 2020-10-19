from utils import *
from datetime import datetime

def step(parameters):
    add_model(parameters)
    model_id = get_model_id(parameters)
    evaluate_model(model_id)
    pass

parameters = {
    'fcst_horizon' : 4,
    'history_length' : 730,
    'lgbm_params':{
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
    #         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 300,
        'num_leaves': 64,
        "min_data_in_leaf": 100,
        'seed': 0,
        'n_jobs': -1
    }
}

# print(f"Started at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 300
# step(parameters)

# print(f"Finished model 1 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 500
# step(parameters)

# print(f"Finished model 2 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 800
# step(parameters)

# print(f"Finished model 3 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 300
# parameters['lgbm_params']['sub_feature'] = 0.7
# step(parameters)

# print(f"Finished model 4 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 300
# parameters['lgbm_params']['sub_feature'] = 0.3
# step(parameters)

# print(f"Finished model 5 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 300
# parameters['lgbm_params']['sub_feature'] = 1.0
# parameters['lgbm_params']['num_leaves'] = 16
# step(parameters)

# print(f"Finished model 6 at {datetime.now()}")

# parameters['lgbm_params']['num_iterations'] = 300
# parameters['lgbm_params']['sub_feature'] = 1.0
# parameters['lgbm_params']['num_leaves'] = 256
# step(parameters)

# print(f"Finished all 7 models at {datetime.now()}")



# store_ids = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
# i = 1
# for store_id in store_ids:
#     print (f'Step {i} out of {len(store_ids)} started at {datetime.now()}')
#     # value_filter = {'state_id' : [state_id], 'cat_id' : [cat_id]}
#     value_filter = {'store_id' : [store_id]}
#     abt = create_abt(value_filter=value_filter)
#     abt.to_pickle(f'./work/abt_{store_id}.pkl')
#     # abt.to_csv(f'./work/baseline_abt_{store_id}.csv')
#     abt = None
#     i += 1
# print (f'Finished creating abt at {datetime.now()}')


parameters = {
    'fcst_horizon' : 4,
    'history_length' : 730,
    'history_start' : 35,
    'lgbm_params':{
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
    #         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "metric": ["rmse"],
        'verbosity': 1,
        'num_iterations' : 300,
        'num_leaves': 16,
        "min_data_in_leaf": 100,
        'seed': 0,
        'n_jobs': -1
    }
}
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
get_val_fcst(model_id)
prepare_val_submission(model_id)

print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')

parameters['history_length'] = 1100
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
# prepare_val_submission(model_id)

print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')

parameters['history_length'] = 730
parameters['lgbm_params']['num_iterations'] = 500
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
get_val_fcst(model_id)
prepare_val_submission(model_id)

print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')

parameters['history_length'] = 730
parameters['lgbm_params']['num_iterations'] = 300
parameters['lgbm_params']['num_leaves'] = 64
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
get_val_fcst(model_id)
prepare_val_submission(model_id)

print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')

parameters['history_length'] = 1100
parameters['lgbm_params']['num_iterations'] = 500
parameters['lgbm_params']['num_leaves'] = 16
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
# prepare_val_submission(model_id)


print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')

parameters['history_length'] = 730
parameters['lgbm_params']['num_iterations'] = 300
parameters['lgbm_params']['num_leaves'] = 16
parameters['lgbm_params']['min_data_in_leaf'] = 10000
add_model(parameters)
model_id = get_model_id(parameters)
evaluate_model(model_id)
get_val_fcst(model_id)
prepare_val_submission(model_id)

print (f'Finished creating creating forecast for model {model_id} at {datetime.now()}')