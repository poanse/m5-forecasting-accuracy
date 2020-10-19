from utils import *
from utils_recursive import *


# parameters = {
#     'fcst_horizon' : None,
#     'history_length' : 730,
#     'history_start' : 35,
#     'recursive' : True,
#     'by_variable' : 'cat_id',
#     'remove_features' : ['lag1', 'lag7', 'id'],
#     'lgbm_params':{
#         "objective" : "poisson",
#         "metric" :"rmse",
#         "force_row_wise" : True,
#         "learning_rate" : 0.05,
#         "sub_feature" : 0.5,
#         "sub_row" : 0.75,
#         "bagging_freq" : 1,
#         "lambda_l2" : 0.1,
#         'verbosity': 1,
#         'num_iterations' : 300,
#         'num_leaves': 16,
#         "min_data_in_leaf": 100,
#         'seed': 0,
#         'n_jobs': -1
#     }
# }

# add_model(parameters)
# model_id = get_model_id(parameters)
# train_model_recursive(model_id, months=[0])
# predict_and_prepare_submit_recursive(model_id, month=0)



# parameters = {
#     'fcst_horizon' : None,
#     'history_length' : 730,
#     'history_start' : 35,
#     'recursive' : True,
#     'by_variable' : 'cat_id',
#     'remove_features' : ['lag1', 'lag7', 'id', 'cat_id'],
#     'lgbm_params':{
#         "objective" : "poisson",
#         "metric" :"rmse",
#         "force_row_wise" : True,
#         "learning_rate" : 0.05,
#         "sub_feature" : 0.5,
#         "sub_row" : 0.75,
#         "bagging_freq" : 1,
#         "lambda_l2" : 0.1,
#         'verbosity': 1,
#         'num_iterations' : 600,
#         'num_leaves': 63,
#         "min_data_in_leaf": 500,
#         'seed': 0,
#         'n_jobs': -1
#     }
# }

# add_model(parameters)
# model_id = get_model_id(parameters)
# train_model_recursive(model_id, months=[0])
# predict_and_prepare_submit_recursive(model_id, month=0)



# parameters = {
#     'fcst_horizon' : None,
#     'history_length' : 730,
#     'history_start' : 35,
#     'recursive' : True,
#     'by_variable' : 'cat_id',
#     'remove_features' : ['lag1', 'lag7','lag14','year','first_sale_day', 'id', 'cat_id'],
#     'lgbm_params':{
#         "objective" : "poisson",
#         "metric" :"rmse",
#         "force_row_wise" : True,
#         "learning_rate" : 0.05,
#         "sub_feature" : 0.5,
#         "sub_row" : 0.75,
#         "bagging_freq" : 1,
#         "lambda_l2" : 0.1,
#         'verbosity': 1,
#         'num_iterations' : 600,
#         'num_leaves': 2**8 - 1,
#         "min_data_in_leaf": 500,
#         'seed': 0,
#         'n_jobs': 2
#     }
# }

# add_model(parameters)
# model_id = get_model_id(parameters)
# train_model_recursive(model_id, months=[0])
# predict_and_prepare_submit_recursive(model_id, month=0)


parameters = {
    'fcst_horizon' : None,
    'history_length' : 730,
    'history_start' : 35,
    'recursive' : True,
    'by_variable' : 'cat_id',
    'remove_features' : ['year','first_sale_day', 'id', 'cat_id'],
    'lgbm_params':{
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.05,
        "sub_feature" : 0.5,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : 600,
        'num_leaves': 2**9 - 1,
        "min_data_in_leaf": 500,
        'seed': 0,
        'n_jobs': -1
    }
}

# add_model(parameters)
# model_id = get_model_id(parameters)
model_id = "0029"
train_model_recursive(model_id, months=[-1])
predict_and_prepare_submit_recursive(model_id, month=-1)
