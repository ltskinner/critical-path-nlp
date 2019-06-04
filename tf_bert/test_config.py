

from configs.config_squad import ConfigSQuAD


def test_config_squad():
    # Base BERT model folder path
    base_model_folder_path = "C:/Users/Angus/models/uncased_L-12_H-768_A-12"

    # Standard BERT config file name
    name_of_config_json_file = "bert_config.json"

    # Standard BERT vocab file name
    name_of_vocab_file = "vocab.txt"

    # Folder for output to be written to
    output_folder_path = base_model_folder_path + "/trained"

    # SQuAD test path
    squad_test_path = "/placeholder.json"

    def test_verbose():
        FLAGS = None
        Flags = ConfigSQuAD()
        Flags.set_tpu_gpu()
        Flags.set_run_configs(is_squad_v2=True)

        Flags.set_task(
            # do_train=True,
            do_predict=True)

        Flags.set_paths(
            bert_config_file=name_of_config_json_file,
            bert_vocab_file=name_of_vocab_file,
            bert_output_dir=output_folder_path,

            # file_to_train=squad_10_path, # Need to figure out path vs folder
            file_to_predict=squad_test_path)

        Flags.set_training_params(
            # max_seq_length=128,
            max_answer_length=45)

        Flags.validate_flags_and_config()

        # Show tests
        FLAGS = Flags.flags.FLAGS
        print(FLAGS.bert_config_file)
        print(FLAGS.max_seq_length)
        print(FLAGS.max_answer_length)
        
    def test_concise():
        FLAGS = None
        Flags2 = ConfigSQuAD()
        Flags2.use_defaults()
        Flags2.set_task(do_predict=True)

        Flags2.set_paths(
            bert_config_file=name_of_config_json_file,
            bert_vocab_file=name_of_vocab_file,
            bert_output_dir=output_folder_path,
            file_to_predict=squad_test_path)
        Flags2.validate_flags_and_config()

        FLAGS = Flags2.flags.FLAGS
        print("concise -->")
        print(FLAGS.bert_config_file)
        print(FLAGS.max_seq_length)
        print(FLAGS.max_answer_length)

    test_concise()
    test_verbose()

test_config_squad()
