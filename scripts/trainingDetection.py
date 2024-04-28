from utils import *

# FACE TRACKER CLASS

class FaceTracker(Model): 
    def __init__(self, facetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

# LOAD AUGMENTED DATA

train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False).map(preprocess_image)
test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False).map(preprocess_image)
val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False).map(preprocess_image)

# LOAD LABELS

train_labels = load_dataset('train')
test_labels = load_dataset('test')
val_labels = load_dataset('val')

# CREATION DATASET

train = prepare_dataset(train_images, train_labels, 5000)
test = prepare_dataset(test_images, test_labels, 1300)
val = prepare_dataset(val_images, val_labels, 1000)

print('Shape del los elementos del train:', train.as_numpy_iterator().next()[0].shape)
print('Contenido de los elementos del train:\n', train.as_numpy_iterator().next()[1])

# NEURAL NETWORK 

vgg = VGG16(include_top=False)
facetracker = build_model()

# OPTIMIZER AND LR

batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# TRAIN

model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

facetracker.save('facetracker.keras')

print('Facetracker saved as facetracker.keras')