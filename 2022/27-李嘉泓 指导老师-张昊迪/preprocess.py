from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import fire

from dataset.outcome import OutcomeDiagnosesDataset, collate_batch
import utils


class OutcomeDiagnosesPreprocessing:

    @staticmethod
    def load_model_and_tokenizer(pretrained_model, hidden_size, seed):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        bert = AutoModel.from_pretrained(pretrained_model)

        bert_hidden_size = bert.config.hidden_size

        linear_layer = None
        if bert_hidden_size != hidden_size:
            # reset the seed to make sure linear layer is the same as in preprocessing
            pl.utilities.seed.seed_everything(seed=seed)
            linear_layer = nn.Linear(bert_hidden_size, hidden_size)

        return tokenizer, bert, linear_layer

    @staticmethod
    def get_train_dataloader(train_file, tokenizer, max_length, all_labels_path, batch_size):
        train_dataset = OutcomeDiagnosesDataset(train_file, tokenizer, max_length=max_length,
                                                all_codes_path=all_labels_path)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       collate_fn=collate_batch,
                                                       batch_size=batch_size,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       shuffle=False)
        return train_dataloader

    def save_vectors_per_sample(self,
                                train_file,
                                all_labels_path,
                                output_path,
                                pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                max_length=512,
                                batch_size=20,
                                hidden_size=256,
                                use_cuda=True,
                                seed=7):
        tokenizer, bert, linear_layer = self.load_model_and_tokenizer(pretrained_model, hidden_size, seed)

        train_dataloader = self.get_train_dataloader(train_file, tokenizer, max_length, all_labels_path, batch_size)

        if use_cuda:
            bert = bert.cuda()

        with torch.no_grad():

            sample_dict = {}
            bert.eval()
            for i, batch in enumerate(train_dataloader):
                bert_vector_per_sample, _ = utils.get_bert_vectors_per_sample(batch,
                                                                              bert,
                                                                              use_cuda,
                                                                              linear=linear_layer)
                batch_dict = dict(zip(batch["sample_ids"], bert_vector_per_sample))
                sample_dict = {**sample_dict, **batch_dict}

        torch.save(sample_dict, f"{output_path}.pt")

    def multiple_prototype_vectors_from_centroids(self,
                                                  train_file,
                                                  all_labels_path,
                                                  output_path,
                                                  sample_vector_path,
                                                  pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                  max_length=512,
                                                  batch_size=20,
                                                  hidden_size=256,
                                                  seed=7
                                                  ):
        tokenizer, _, _ = self.load_model_and_tokenizer(pretrained_model, hidden_size, seed)

        train_dataloader = self.get_train_dataloader(train_file, tokenizer, max_length, all_labels_path, batch_size)
        train_dataset = train_dataloader.dataset

        vectors = torch.load(sample_vector_path)

        centroids = {}
        for label in train_dataset.labels:
            samples_for_code = train_dataset.data[train_dataset.data[label] == 1].id.tolist()

            sample_vectors_for_code = [vectors[sample].tolist() for sample in samples_for_code]

            if len(sample_vectors_for_code) == 0:
                centroids[label] = [torch.rand(hidden_size).tolist()]

            elif len(sample_vectors_for_code) == 1:
                centroids[label] = [sample_vectors_for_code[0]]

            elif len(sample_vectors_for_code) > 1:
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5).fit(sample_vectors_for_code)

                clusters = []
                for j in range(clustering.labels_.max() + 1):
                    cluster_samples = np.where(clustering.labels_ == j)[0]
                    cluster_vectors = [sample_vectors_for_code[ind] for ind in cluster_samples]
                    clusters.append(cluster_vectors)

                code_centroids = [list(np.mean(cluster, axis=0)) for cluster in clusters]
                centroids[label] = code_centroids

        torch.save(centroids, f"{output_path}.pt")

    def prototype_vectors_from_centroids(self,
                                         train_file,
                                         all_labels_path,
                                         output_path,
                                         pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                         max_length=512,
                                         batch_size=20,
                                         hidden_size=256,
                                         use_cuda=True,
                                         seed=7
                                         ):
        tokenizer, bert, linear_layer = self.load_model_and_tokenizer(pretrained_model, hidden_size, seed)

        train_dataloader = self.get_train_dataloader(train_file, tokenizer, max_length, all_labels_path, batch_size)
        train_dataset = train_dataloader.dataset
        num_classes = len(train_dataset.labels)

        n_per_class = torch.zeros(num_classes)
        sum_per_class = torch.zeros([num_classes, hidden_size])

        if use_cuda:
            bert = bert.cuda()
            n_per_class = n_per_class.cuda()
            sum_per_class = sum_per_class.cuda()

        with torch.no_grad():

            bert.eval()
            for i, batch in enumerate(train_dataloader):
                target_tensors = torch.tensor(batch["targets"])
                if use_cuda:
                    target_tensors = target_tensors.cuda()

                bert_vector_per_sample, token_vectors = utils.get_bert_vectors_per_sample(batch,
                                                                                          bert,
                                                                                          use_cuda,
                                                                                          linear=linear_layer)

                # get tensor of shape batch_size x num_classes x dim
                masked_vectors_per_class = torch.einsum('ik,il->ilk', bert_vector_per_sample, target_tensors)

                # sum into one vector per prototype. shape: num_classes x dim
                sum_per_class = torch.add(sum_per_class, masked_vectors_per_class.sum(dim=0).detach())

                n_per_class += target_tensors.sum(dim=0).detach()

                # prevent zero division
                n_per_class[n_per_class == 0] = 1

                # divide summed vectors by n class occurrences
                averaged_vectors_per_prototype = torch.div(sum_per_class, n_per_class.unsqueeze(1))

                vectors_per_prototype = [
                    v.tolist() if not (v.max() == 0 and v.min() == 0) else torch.rand(hidden_size).tolist() for v in
                    averaged_vectors_per_prototype]

                prototype_map = {label: [vectors_per_prototype[i]] for i, label in enumerate(train_dataset.labels)}

                # cache prototype vectors
                torch.save(prototype_map, f"{output_path}.pt")

    def attention_vectors_from_tf_idf(self,
                                      train_file,
                                      all_labels_path,
                                      output_path,
                                      pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                      max_length=512,
                                      batch_size=20,
                                      hidden_size=256,
                                      use_cuda=True,
                                      seed=7):
        tokenizer, bert, linear_layer = self.load_model_and_tokenizer(pretrained_model, hidden_size, seed)

        train_dataloader = self.get_train_dataloader(train_file, tokenizer, max_length, all_labels_path, batch_size)
        train_dataset = train_dataloader.dataset
        num_classes = len(train_dataset.labels)

        def tokenize_tf_idf(text_document):
            tokenized = tokenizer(text_document)

            return tokenized.encodings[0].tokens[1:-1]

        vectorizer = TfidfVectorizer(tokenizer=tokenize_tf_idf, max_features=30_000, max_df=0.7,
                                     stop_words="english")

        # fit TF-IDF vectorizer on whole train set
        tf_idf = vectorizer.fit_transform(train_dataset.data.text.tolist())

        features = vectorizer.get_feature_names()

        n_att_per_class = torch.zeros(num_classes)
        sum_att_per_class = torch.zeros([num_classes, hidden_size])

        if use_cuda:
            bert = bert.cuda()
            n_att_per_class = n_att_per_class.cuda()
            sum_att_per_class = sum_att_per_class.cuda()

        with torch.no_grad():

            bert.eval()
            for i, batch in enumerate(train_dataloader):
                target_tensors = torch.tensor(batch["targets"])
                if use_cuda:
                    target_tensors = target_tensors.cuda()

                bert_vector_per_sample, token_vectors = utils.get_bert_vectors_per_sample(batch,
                                                                                          bert,
                                                                                          use_cuda,
                                                                                          linear=linear_layer)
                all_relevant_tokens = []
                for j, sample in enumerate(batch["tokens"]):

                    global_sample_ind = train_dataloader.dataset.data.id.tolist().index(batch["sample_ids"][j])
                    tf_idf_sample = tf_idf[global_sample_ind]
                    relevant_tokens_sample = []
                    for k in range(batch["input_ids"].shape[1]):
                        if k < len(sample):
                            token = sample[k]
                            if token in features:
                                token_ind = features.index(token)
                                if token_ind in tf_idf_sample.indices:
                                    tf_idf_ind = np.where(tf_idf_sample.indices == token_ind)[0][0]
                                    token_value = tf_idf_sample.data[tf_idf_ind]
                                    if token_value > 0.05:
                                        relevant_tokens_sample.append(1)
                                        continue
                        relevant_tokens_sample.append(0)
                    all_relevant_tokens.append(relevant_tokens_sample)

                all_relevant_tokens = torch.tensor(all_relevant_tokens)
                if use_cuda:
                    all_relevant_tokens = all_relevant_tokens.cuda()

                relevant_tokens = torch.einsum('ik,ikl->ikl', all_relevant_tokens, token_vectors)

                mean_over_relevant_tokens = relevant_tokens.mean(dim=1)

                # get tensor of shape batch_size x num_classes x dim
                masked_att_vectors_per_sample = torch.einsum('ik,il->ilk', mean_over_relevant_tokens,
                                                             target_tensors)

                # sum into one vector per class. shape: num_classes x dim
                sum_att_per_class = torch.add(sum_att_per_class, masked_att_vectors_per_sample.sum(dim=0)
                                              .detach())

                n_att_per_class += target_tensors.sum(dim=0).detach()

        # prevent zero division
        n_att_per_class[n_att_per_class == 0] = 1

        # divide summed vectors by n class occurrences
        averaged_att_vectors_per_class = torch.div(sum_att_per_class, n_att_per_class.unsqueeze(1))

        if use_cuda:
            averaged_att_vectors_per_class = averaged_att_vectors_per_class.cuda()

        # cache prototype vectors
        torch.save(averaged_att_vectors_per_class.detach(), f"{output_path}.pt")


if __name__ == '__main__':
    fire.Fire(OutcomeDiagnosesPreprocessing)
