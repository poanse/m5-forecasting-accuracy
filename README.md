# m5-forecasting-accuracy

[Соревнование на Kaggle по прогнозированию продаж WALMART](https://www.kaggle.com/c/m5-forecasting-accuracy/overview)

## Содержание репозитория:

*fcst_weekly.ipynb* - понедельный прогноз.

*disagg_coefs.ipynb* - дизагрегация понедельного прогноза до подневного: для каждого месяца считалась доля понедельников, вторников и тд в суммарных продажах.
Заброшена из-за сложного распределения продаж между днями недели, в том числе плохо учитывались snap дни.

*fcst_daily.ipynb* - подневный прогноз.

Затем было решено организовать работу следующим образом:
1) Функции для создание витрины с фичами находятся в *utils.py*
2) Запуск обучения модели происходит в скрипте *step.py*, где parameters - конфиг (dict) с lgbm-параметрами, разрезом (по штатам и/или департаментам и тп) и др.
  *  add_model(parameters)
  *  model_id = get_model_id(parameters)
  *  evaluate_model(model_id)
  *  get_val_fcst(model_id)
  *  prepare_val_submission(model_id)
3) Информация об использованных моделях хранится в *master.csv*, где планировалось также хранить ошибку прогноза за различные периоды

Также был использован рекурсивный подход к прогнозированию (который показал более высокую точность в публичном рейтинге) с аналогичной структурой: файлы *step_recursive.py* и *utils_recursive.py*
