from tydi_canine import char_splitter, run_tydi_lib
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Arguments for IO path
    parser.add_argument("--output_dir",
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_input_dir",
                        help="Precomputed input ids for training.")
    parser.add_argument("--state_dict_path",
                        help="model state dict path i.e. model.pdparams.")
    parser.add_argument("--checkout_steps", type=int, default=40000, help="number of steps for save checkout point")
    parser.add_argument("--predict_file",
                        help="TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
                             "Used only for `--do_predict`.")
    parser.add_argument("--precomputed_predict_file",
                        help="TyDi tf.Example records for predictions, created separately by "
                             "`prepare_tydi_json_data.py` Used only for `--do_predict`.")
    parser.add_argument("--output_prediction_file",
                        help="Where to print predictions in TyDi prediction format, to be passed to"
                             "tydi_eval.py.")

    # Arguments for tydi evaluation
    parser.add_argument("--candidate_beam", type=int,
                        help="How many wordpiece offset to be considered as boundary at inference time.")
    parser.add_argument("--max_seq_length", type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                             "Sequences longer than this will be truncated, and sequences shorter "
                             "than this will be padded.")
    parser.add_argument("--max_question_length", type=int,
                        help="The maximum number of tokens for the question. Questions longer than "
                             "this will be truncated to this length.")
    parser.add_argument("--max_to_predict", type=int,
                        help="Maximum number of examples to predict (for debugging). "
                             "`None` or `0` will disable this and predict all.")
    parser.add_argument("--max_position", type=int,
                        help="Maximum passage position for which to generate special tokens.")

    # Arguments for running mode
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')
    parser.add_argument("--do_eval_file_construct", default=False, action='store_true')

    # Arguments for Fine-tuning
    parser.add_argument("--train_batch_size", type=int, )
    parser.add_argument("--predict_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--warmup_proportion", type=float, help="Proportion of steps for linear warmup, 0.1 for 10%.")
    parser.add_argument("--logging_steps", type=int, )
    parser.add_argument("--max_answer_length", type=int, default=2048,
                        help="An upper bound on the number of subword pieces "
                             "that a generated answer may contain. This is needed because the start and "
                             "end predictions are not conditioned on one another.")
    parser.add_argument("--scale_loss", type=float, default=4096, help="The value of scale_loss for fp16.", )
    parser.add_argument("--fp16", default=False, action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--dev_split_ratio", type=float, default=0.002)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps.")
    parser.add_argument("--seed", type=int, default=2020, help="seed.")
    args = parser.parse_args()
    return args


class CanineTyDiRunner(run_tydi_lib.TyDiRunner):
    """CANINE version of TyDiRunner."""

    def __init__(self, args):
        super(CanineTyDiRunner, self).__init__(
            # IO args
            output_dir=args.output_dir,
            train_input_dir=args.train_input_dir,
            state_dict_path=args.state_dict_path,
            predict_file=args.predict_file,
            precomputed_predict_file=args.precomputed_predict_file,
            output_prediction_file=args.output_prediction_file,

            # Tydi Task Evaluation args
            candidate_beam=args.candidate_beam,
            max_position=args.max_position,
            max_to_predict=args.max_to_predict,

            # running mode args
            do_train=args.do_train,
            do_predict=args.do_predict,
            do_file_construct=args.do_eval_file_construct,

            # fine-tuning args
            checkout_steps=args.checkout_steps,
            max_seq_length=args.max_seq_length,
            train_batch_size=args.train_batch_size,
            predict_batch_size=args.predict_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_proportion=args.warmup_proportion,
            logging_steps=args.logging_steps,
            dev_split_ratio=args.dev_split_ratio,
            max_answer_length=args.max_answer_length,
            fp16=args.fp16,
            scale_loss=args.scale_loss,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed
        )
        self.validate_args_or_throw()

    def validate_args_or_throw(self):
        """Validate the input args or throw an exception."""
        if self.output_dir is None:
            raise ValueError("output_dir is required.")

        if not self.do_train and not self.do_predict and not self.do_file_construct:
            raise ValueError("At least one of `{do_train,do_predict,do_file_construct}` must be True.")

        if self.do_train:
            if not self.train_input_dir:
                raise ValueError("If `do_train` is True, then `train_input_dir` "
                                 "must be specified.")
            if not self.train_batch_size:
                raise ValueError("If `do_train` is True, then `train_batch_size` "
                                 "must be specified.")
            if not self.learning_rate:
                raise ValueError("If `do_train` is True, then `learning_rate` "
                                 "must be specified.")
            if not self.num_train_epochs:
                raise ValueError("If `do_train` is True, then `num_train_epochs` "
                                 "must be specified.")
            if not self.warmup_proportion:
                raise ValueError("If `do_train` is True, then `warmup_proportion` "
                                 "must be specified.")
        else:
            if self.train_batch_size is None:
                self.train_batch_size = 1

        if self.do_predict:
            if not self.predict_file:
                raise ValueError("If `do_predict` is True, "
                                 "then `predict_file` must be specified.")
            if not self.max_answer_length:
                raise ValueError("If `do_predict` is True, "
                                 "then `max_answer_length` must be specified.")
            if not self.candidate_beam:
                raise ValueError("If `do_predict` is True, "
                                 "then `candidate_beam` must be specified.")
            if not self.predict_batch_size:
                raise ValueError("If `do_predict` is True, "
                                 "then `predict_batch_size` must be specified.")
            if not self.output_prediction_file:
                raise ValueError("If `do_predict` is True, "
                                 "then `output_prediction_file` must be specified.")

            if not self.precomputed_predict_file:
                raise ValueError("`precomputed_predict_file` must be provided.")


def main():
    args = get_args()
    CanineTyDiRunner(args).run()


if __name__ == "__main__":
    main()
