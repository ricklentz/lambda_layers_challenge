import tensorflow as tf
import tensorflow_hub as hub

def get_features(list_of_text):
	# s3.amazonaws.com/rocket-ml/15a90830a567dd734b2ae775fd3c5d360b081ca8.tar.gz
    embed = hub.Module("https://s3.amazonaws.com/rocket-ml/15a90830a567dd734b2ae775fd3c5d360b081ca8.tar.gz")
    with tf.Session() as session:
       session.run([tf.global_variables_initializer(), tf.tables_initializer()])
       message_embeddings = session.run(embed(list_of_text,signature="default",as_dict=False))
    return message_embeddings


list_of_text = ['The quick brown fox jumped over the lazy dog.','Nintey-nine problems but lambda is not one.','Check out Klaus Seilers Medium article: Pure serverless machine learning inference with AWS Lambda and Layers.']
print( get_features(list_of_text) )