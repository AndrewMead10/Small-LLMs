import torch
from torch import nn
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import math


class CalmModel(nn.Module):
    def __init__(self, model, calm_lambda, calm_temp):
        super().__init__()
        self.model = model

        self.early_exit_decay = self.calc_early_exit_decay(calm_lambda, calm_temp)
        self.calm_lambda = calm_lambda
        self.calm_temp = calm_temp

        self.predicted_words = None
        self.predicted_words_prob = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        index: torch.Tensor = None,
        rope: torch.Tensor = None,
        show_layer_probs: bool = True,
        max_length: int = None,
    ) -> torch.Tensor:
        bs, seq_len = x.shape

        if max_length is not None and max_length != len(self.early_exit_decay):
            self.early_exit_decay = self.calc_early_exit_decay(
                self.calm_lambda, self.calm_temp, max_length
            )

        x = self.model.embedding(x)

        if show_layer_probs:
            pred_words = torch.zeros(
                (bs, self.model.n_blocks), device=self.model.device
            )
            pred_word_probs = torch.zeros(
                (bs, self.model.n_blocks), device=self.model.device
            )

        for i, block in enumerate(self.model.blocks):
            x = block(
                x,
                kv_cache=self.model.kv_cache[i],
                mask=mask,
                index=index,
                rope=rope,
            )

            expected_out = self.model.lm_head(x)
            probs = torch.softmax(expected_out, dim=-1)

            if probs.shape[1] > 1:
                probs = probs[:, -1, :]

            pred_word_prob = torch.max(probs, dim=-1).values

            if show_layer_probs:
                pred_word = torch.argmax(probs, dim=-1)

                pred_words[:, i] = pred_word.flatten()
                pred_word_probs[:, i] = pred_word_prob.flatten()

            prob_greater_than = self.early_exit_decay[i]

            if pred_word_prob > prob_greater_than:
                # fill in kv caches for the remaining blocks
                print("early exit at block {} for index {}".format(i, index))
                for j in range(i + 1, self.model.n_blocks):
                    self.model.blocks[j].mha(
                        x,
                        kv_cache=self.model.kv_cache[j],
                        mask=mask,
                        index=index,
                        rope=rope,
                    )

                if show_layer_probs:
                    self.calm_tracking(pred_words, pred_word_probs)

                return expected_out

        if show_layer_probs:
            self.calm_tracking(pred_words, pred_word_probs)

        x = self.model.lm_head(x)

        return x

    def calm_tracking(self, pred_word, pred_word_prob):
        if self.predicted_words is None:
            self.predicted_words = pred_word
            self.predicted_words_prob = pred_word_prob
        else:
            self.predicted_words = torch.cat((self.predicted_words, pred_word), dim=0)
            self.predicted_words_prob = torch.cat(
                (self.predicted_words_prob, pred_word_prob), dim=0
            )

    def show_calm(self):
        hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

        for i in range(len(self.predicted_words)):
            # plot the prob of each word over time and when it matched the output word
            plt.plot(self.predicted_words_prob[i])
            first_word_index = torch.where(
                self.predicted_words[i] == self.predicted_words[i][-1]
            )[0][0]

            plt.scatter(
                x=first_word_index,
                y=self.predicted_words_prob[i][first_word_index],
                color="r",
                marker="x",
                s=100,
            )

            # decode the output word
            decoded_word = hf_tokenizer.decode(
                self.predicted_words[i][-1].to(torch.int64), skip_special_tokens=True
            )

            # Add the decoded word as a label to the plot
            plt.annotate(
                decoded_word,
                (first_word_index, self.predicted_words_prob[i][-1]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )

        plt.show()

    def calc_early_exit_decay(self, calm_lambda, calm_temp, max_length=None):
        if max_length is None:
            max_length = self.model.max_seq_len
        vals = []
        for i in range(self.model.n_blocks):
            vals.append(0.9 * calm_lambda + 0.1 * math.exp(-calm_temp * i / max_length))

        return vals
