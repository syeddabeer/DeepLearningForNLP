# depends on 6.33
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data, 
	lookback=lookback, 
	delay=delay, 
	min_index=0, 
	max_index=200000,
	shuffle=True,
	step=step,
	batch_size=batch_size)
val_gen = generator(float_data, 
	lookback=lookback, 
	delay=delay, 
	min_index=200001, 
	max_index=300000,
	shuffle=True,
	step=step,
	batch_size=batch_size)
test_gen = generator(float_data, 
	lookback=lookback, 
	delay=delay, 
	min_index=300001, 
	max_index=None,
	shuffle=True,
	step=step,
	batch_size=batch_size)

val_steps=(300000-200001-lookback)
test_steps = (len(float_data)-300001-lookback)