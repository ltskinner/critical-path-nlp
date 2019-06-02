# Load naked model
from xxx.trainer import BertModel
from xxx.trainer import load_naked_gpt2

# Format data for various tasks
from xxx.tasks import qa_output
from xxx.tasks import classification_output
from xxx.tasks import word_embedding_output
from xxx.tasks import document_embedding_output
from xxx.tasks import language_generator

naked_bert = BertModel()

naked_bert = BertModel.load_naked_bert('file/path') # Configuration file will live next to model
naked_gpt2 = load_naked_gpt2('file/path')

naked_bert.set_output_task(qa_output)
naked_gpt2.set_output_task(language_generator)

from dataloader import qa_data_handle
from dataloader import classification_data_handle
from dataloader import word_embedding_handler
from dataloader import document_embedding_handler
from dataloader import language_generator_data_handle

# Want to think about how to divy into test train split
qa_data = qa_data_handle(qa_data_source, training=True)
language_generator_data = language_generator_data_handle(
                            language_generator_data_source, training=True)


from xxx.trainer import bert_training_params
from xxx.trainer import gpt2_training_params

print(bert_training_params) # Have a repr with notes on good default vals
print(gpt2_training_params)

bert_training_params.set_param(name="param", value="")
gpt2_training_params.set_param(name="param", value="")

naked_bert.train_on_params(bert_training_params, qa_data)
naked_gpt2.train_on_params(bert_training_params, language_generator_data)

naked_bert.save_trained('/file/path/name')
naked_gpt2.save_trained('/file/path/name')


from xxx.predictor import BertModel
from xxx.predictor import trained_gpt2

bert_for_qa = BertModel()
bert_for_qa = bert_for_qa.load_trained_bert('/file/path/name') # output="") # ?
gpt2_for_language_generation = trained_gpt2.load('/file/path/name')

from dataloader import qa_data_handle
from dataloader import language_generator_data_handle

qa_testing_data = qa_data_handle(qa_data_source, predict=True)
language_generator_testing_data = language_generator_data_handle(
                            language_generator_data_source, predict=True)

qa_results = bert_for_qa.evaluate(qa_testing_data)
language_generator_results = gpt2_for_language_generation.evaluate(
                                            language_generator_testing_data)

qa_results.display() # Specific display function for qa
language_generator_results.display() # Specific display function for lang_gen



