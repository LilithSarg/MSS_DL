import tensorflow as tf
from models.DNN import *

# Datasets
tf.app.flags.DEFINE_string('train_spec_dir', '/content/gdrive/My Drive/project_folder/DL_project/MUSDB18-HQ/spectograms/train', 'Training spectograms data directory.')
tf.app.flags.DEFINE_string('val_spec_dir', '/content/gdrive/My Drive/project_folder/DL_project/MUSDB18-HQ/spectograms/val', 'Validation spectograms data directory.')
tf.app.flags.DEFINE_string('test_spec_dir', '/content/gdrive/My Drive/project_folder/DL_project/MUSDB18-HQ/spectograms/val', 'Testing spectograms data directory.')

tf.app.flags.DEFINE_boolean('train', True, 'whether to train the network')
tf.app.flags.DEFINE_integer('num_epochs', 3, 'epochs to train')
tf.app.flags.DEFINE_integer('train_batch_size', 100, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 100, 'number of elements in a validation batch')
tf.app.flags.DEFINE_integer('test_batch_size', 100, 'number of elements in a testing batch')

tf.app.flags.DEFINE_integer('sequence_length', 199, 'Ms occurrences in a second frame.') #height_of_image 
tf.app.flags.DEFINE_float('fft_length', 802, 'Fourier coefficients in a sequence length.') #width of image #should be 401 DONE

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer')

tf.app.flags.DEFINE_integer('display_step', 2, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 2, 'Number of steps we cycle through before validating the model.')

tf.app.flags.DEFINE_string('base_dir', '/content/gdrive/My Drive/project_folder/DL_project/MSS_DL/results', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 300, 'Number of steps we cycle through before saving checkpoint.')
tf.app.flags.DEFINE_integer('max_to_keep', 5, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 51, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('model_name', 'lstm_test', 'name of model')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    model = DNN(
        train_spec_dir=FLAGS.train_spec_dir,
        val_spec_dir=FLAGS.val_spec_dir,
        test_spec_dir=FLAGS.test_spec_dir,
        num_epochs=FLAGS.num_epochs,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        test_batch_size=FLAGS.test_batch_size,
        sequence_length=FLAGS.sequence_length,
        fft_length=FLAGS.fft_length,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name,
    )

    model.create_network()
    print('STATUS -------------> NETWORK CREATED')
    model.initialize_network()
    print('STATUS -------------> NETWORK INITIALIZED')
    if FLAGS.train:
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
        print('STATUS -------------> MODEL TRAINED')
    else:
        model.test_model()
        print('STATUS -------------> MODEL TESTED')


if __name__ == "__main__":
    tf.app.run()
