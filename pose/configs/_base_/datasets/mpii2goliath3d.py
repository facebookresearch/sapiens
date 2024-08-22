from configs._base_.datasets.goliath3d import dataset_info as goliath_info

goliath_info = goliath_info.build()
dataset_info = goliath_info.copy()
dataset_info['dataset_name'] = 'mpii2goliath3d'
