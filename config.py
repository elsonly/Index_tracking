config = {
	'index':'tw50',
	'start':'2006-01-01',
	'end':'2016-01-01',
	'standard':1e-4,
	'validation_ratio':0.1,
	'testing_period':60,
	'window':128,
	'holding_period':20,
	'batch_size':32,
	'continuous_sample':False,

	'learning_rate':1e-5,
	'training_steps':30,#int(4e+4),
	'save_model':True,
	'save_step':1000,
	'print_process':True,
	'early_stop':4,
	
}

