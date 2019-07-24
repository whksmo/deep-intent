model = dm.DSCN(task=6723)
model.compile("adam", "sparse_categorical_crossentropy", metrics=['accuracy'])

print('start training...')
history = model.fit(train_model_input, data[target].values, batch_size=256, epochs=20, verbose=2, validation_data=(test_model_input, test_data[target].values))
model.save('./dscn.h5')
