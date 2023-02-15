import os
import json
import torch
import sys

from subprocess import run
from data.batch import Batch

sys.path.append("../evaluation")
from evaluate_single_dataset import evaluate


def predict(model, data, input_path, raw_input_path, args, logger, output_directory, device, mode="validation", epoch=None):
    model.eval()

    framework, language = args.framework, args.language
    sentences = {}
    with open(input_path, encoding="utf8") as f:
        for line in f.readlines():
            line = json.loads(line)
            line["nodes"], line["edges"], line["tops"] = [], [], []
            line["framework"], line["language"] = framework, language
            sentences[line["id"]] = line

    for i, batch in enumerate(data):
        with torch.no_grad():
            predictions = model(Batch.to(batch, device), inference=True)
            for prediction in predictions:
                for key, value in prediction.items():
                    sentences[prediction["id"]][key] = value

    if epoch is not None:
        output_path = f"{output_directory}/prediction_{mode}_{epoch}_{framework}_{language}.json"
    else:
        output_path = f"{output_directory}/prediction.json"

    with open(output_path, "w", encoding="utf8") as f:
        for sentence in sentences.values():
            json.dump(sentence, f, ensure_ascii=False)
            f.write("\n")
            f.flush()

    run(["./convert.sh", output_path] + (["--node_centric "] if args.graph_mode != "labeled-edge" else []))

    if raw_input_path:
        results = evaluate(raw_input_path, f"{output_path}_converted")
        print(mode, results, flush=True)

        if logger is not None:
            logger.log_evaluation(results, mode, epoch)

        return results["sentiment_tuple/f1"]
