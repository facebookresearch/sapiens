from configs._base_.datasets.goliath import dataset_info as goliath_info

goliath_info = goliath_info.build()
dataset_info = goliath_info.copy()
dataset_info['dataset_name'] = 'aic2goliath'
