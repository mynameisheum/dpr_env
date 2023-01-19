#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import tokenizers
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils

#####################################
from transformers import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

shuffle_list = []

#####################################
def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out

    def create_biencoder_input(
        self,
        epoch : None,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        passage_max_length: int = 0,
        num_idx: int = 0,
        Method = None,
        neg_in_neg:int = 0,

    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        # print(Method)


        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)


            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]



            if Method == 'random':  # random
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

                neg_ctxs = neg_ctxs[0:num_other_negatives]  # 2
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]  # 1

                all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs

            if Method == 'top-1': # Top-1

                neg_ctxs = neg_ctxs[0:num_other_negatives]  # 2
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives] # 1

                all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs

            if Method == 'uniform': # Uniform

                init_epoch = 0
                # print('show me the money')
                # nuiform 방식의 hard 1개 사용s
                # neg_ctxs = neg_ctxs[ epoch - init_epoch]  # num of hard negative = 1

                # uniform 방식의 hard n개 사용 inbatch 방식중 1번째 : long uniform / n = num_hard_negatives, num_other_negatives
                neg_ctxs = neg_ctxs[ num_other_negatives * (epoch - init_epoch ) : num_other_negatives * (epoch - init_epoch + 1) ]  # number of hard negative = n and 30 번 다 확인
                # hard_neg_ctxs = hard_neg_ctxs[num_hard_negatives * (epoch - init_epoch): num_hard_negatives * (epoch - init_epoch + 1)]  # number of hard negative = n and 30 번 다 확인

                # uniform 방식의 hard n개 사용 inbatch 방식중 2번째 : uniform + top-1
                # init_epoch = 1
                # neg_ctxs = neg_ctxs * 3
                # neg_ctxs = neg_ctxs[num_other_negatives * (epoch - init_epoch): num_other_negatives * (epoch - init_epoch + 1)]  # num of hard negative = 1

                hard_neg_ctxs = []
                # neg_ctxs = []

                all_ctxs = [positive_ctx] + neg_ctxs
                # all_ctxs = [positive_ctx] + hard_neg_ctxs

            if Method == 'active': # Active
                num_of_hard_neg = 1

                question_tokenizer = tokenizer_bert(question, add_special_tokens=True, padding='max_length',truncation=True, max_length=512,
                               return_tensors='pt').to(torch.device('cuda'))
                # self.ctx_model.eval()
                query_representation = self.question_model(question_tokenizer['input_ids'],
                                                      question_tokenizer['token_type_ids'],
                                                      question_tokenizer['attention_mask']) # [0] = 1,512,768 / [1] = 1,768

                vector_neg_ctx_representation = []
                for neg_ctx in neg_ctxs:
                    neg_ctx_tokenizer = tokenizer_bert(neg_ctx[0], add_special_tokens=True, padding='max_length',truncation=True, max_length=512, # neg[0] = text, neg[1] = title
                                        return_tensors='pt').to(torch.device('cuda'))
                    # print(neg_ctx_tokenizer['input_ids'])
                    # print(neg_ctx_tokenizer['input_ids'][0].shape)

                    if len(neg_ctx_tokenizer['input_ids'][0]) >= 512:
                       neg_ctx_tokenizer['input_ids'][0] = neg_ctx_tokenizer['input_ids'][0][:512]
                       neg_ctx_tokenizer['input_ids'][0][511] = 102
                       neg_ctx_tokenizer['token_type_ids'][0] = neg_ctx_tokenizer['token_type_ids'][0][:512]
                       neg_ctx_tokenizer['attention_mask'][0] = neg_ctx_tokenizer['attention_mask'][0][:512]

                    neg_ctx_representation = self.ctx_model(neg_ctx_tokenizer['input_ids'], neg_ctx_tokenizer['token_type_ids'], neg_ctx_tokenizer['attention_mask'])

                    vector_neg_ctx_representation.append(neg_ctx_representation[1])
                vector_neg_ctx_representation_to_stack = torch.stack(vector_neg_ctx_representation)
                inner_query_neg = torch.matmul(vector_neg_ctx_representation_to_stack, query_representation[1].transpose(1,0))
                top_neg_in_neg_index = torch.topk(inner_query_neg.flatten(), k=num_of_hard_neg, dim=0).indices  # 보통 argmax해서 0이 나오는게 맞지.가장 유사도가 높은순서로 neg 집합을 만들었으니까

                if top_neg_in_neg_index[0] != 0:
                    # print(f'found neg in neg! count : {neg_in_neg}')
                    neg_in_neg += 1
                neg_ctxs = neg_ctxs[top_neg_in_neg_index] # index로 뽑아서 접근하니까 [ : ]처럼 접근하면 안됨
                # if num_of_hard_neg > 1이면  neg_ctxs[top_neg_in_neg_index] +neg_ctxs[top_neg_in_neg_index]+neg_ctxs[top_neg_in_neg_index] 쭉쭉..
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

                all_ctxs = [positive_ctx] + [neg_ctxs] + hard_neg_ctxs
                # self.ctx_model.train()

            ####################################################
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)


            # sample_ctxs_tensors2 = []
            # for ctx in all_ctxs:
            #     temp1 = tokenizer_bert(ctx.title, ctx.text, add_special_tokens=True, padding='max_length' ,max_length=256, return_tensors='pt')['input_ids'][0]
            #     # sample_ctxs_tensors2.append(temp1)
            #     if len(temp1) <= 256:
            #         temp1[255] = 102
            #         sample_ctxs_tensors2.append(temp1)
            #     else:
            #         temp2 = temp1[:256]
            #         temp2[255] = 102
            #         sample_ctxs_tensors2.append(temp2)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(current_ctxs_len + hard_negatives_start_idx,current_ctxs_len + hard_negatives_end_idx,)
                ]
            )

            if query_token: # False
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else: # True
                question_tensors.append(tensorizer.text_to_tensor(question))
                # question_tensors.append(tensorizer.query_to_tensor(text = question, max_length=64))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0) # question_tensor =16 * 4 (positive 1, hard neg1, neg2)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0) # batch size = 16 , questions_tensor = 16

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor, # batch size: 128 token
            question_segments, # batch size : 128 zero
            ctxs_tensor, # batch size : 512 token
            ctx_segments, # batch size : 512 zero
            positive_ctx_indices,
            hard_neg_ctx_indices, # other neg로 했고 hard로는 안해서 0인거임
            "question",
        ) ,num_idx , neg_in_neg

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()

    def create_val_biencoder_input(
            self,
            samples: List[BiEncoderSample],
            tensorizer: Tensorizer,
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
            hard_neg_fallback: bool = True,
            query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx,
                )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))
                # question_tensors.append(tensorizer.query_to_tensor(question, max_length=64))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)  # dot_product(batch size : 768 CLS , batch size*(q+hard) : 768 CLS) = (batch: batch)
        #                                                   # ex) batch = 4, hard= 3 / scores = (4,16)
        # # if RS_sampling :
        # RS_row_1 = torch.cat([scores[0,::4],scores[0,1:4]]).view(1,-1) # [for batch::batch],[for batch, batch*h3-3,batch*h3]
        # RS_row_2 = torch.cat([scores[1,::4],scores[1,5:8]]).view(1,-1)
        # RS_row_3 = torch.cat([scores[2,::4],scores[2,9:12]]).view(1,-1)
        # RS_row_4 = torch.cat([scores[3,::4],scores[3,13:16]]).view(1,-1)
        #
        # RS_sample = torch.cat([RS_row_1, RS_row_2, RS_row_3, RS_row_4]) # ex) batch = 4, hard =3 / RS_sample = (4,7)

        # base line
        # positive_idx_per_question = [0,4,8,12]

        # RS_sample for positive idx
        # positive_idx_per_question = [0, 1, 2, 3]
        # positive_idx_per_question = [0, 1, 2, 3, 4, 5, 6, 7]

        # Robust = self.get_scores(q_vectors, torch.cat([ctx_vectors[0].view(1,-1), ctx_vectors[4].view(1,-1), ctx_vectors[8].view(1,-1), ctx_vectors[12].view(1,-1)])) # (4,768) * (4,768)
        # Semantic_1 = self.get_scores(q_vectors[0], torch.cat([ctx_vectors[1].view(1,-1),ctx_vectors[2].view(1,-1),ctx_vectors[3].view(1,-1)])) # (1, 768) * (3, 768)
        # Semantic_2 = self.get_scores(q_vectors[1], torch.cat([ctx_vectors[5].view(1,-1),ctx_vectors[6].view(1,-1),ctx_vectors[7].view(1,-1)]))  # (1, 768) * (3, 768)
        # Semantic_3 = self.get_scores(q_vectors[2], torch.cat([ctx_vectors[9].view(1,-1),ctx_vectors[10].view(1,-1),ctx_vectors[11].view(1,-1)]))  # (1, 768) * (3, 768)
        # Semantic_4 = self.get_scores(q_vectors[3], torch.cat([ctx_vectors[13].view(1,-1),ctx_vectors[14].view(1,-1),ctx_vectors[15].view(1,-1)]))  # (1, 768) * (3, 768)
        # Semantic = torch.cat([Semantic_1.view(1,-1),Semantic_2.view(1,-1),Semantic_3.view(1,-1),Semantic_4.view(1,-1)]) #(4,3)
        # RS_sample = torch.cat([ Robust, Semantic ], dim=1)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            # baseline
            # scores = scores.view(q_num, -1)
            # RS_sample
            scores = scores.view(q_num,-1)

        softmax_scores = F.log_softmax(scores, dim=1) # batch : batch(q +hard3) # baseline(scores) or our sampling

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device), # to cuda [0,1,2,3,~15]중에 0 index를 분자로, 전체를 분모로 nll 하겠다.
            reduction="mean", # change positive_idex_per_question
        )
        # 이상적이라면 max_idxs = 0,2,4,6,8,10,12,14,16이여야 되지만 역시 조금씩 다르네
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        # 아 그래서 여기서 index로 매칭되는 녀석을 counting하네
        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)
    # query_tensor = tensorizer.query_to_tensor(text, max_length=64)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        # query_tensor_full = tensorizer.query_to_tensor(text, apply_max_len=False, max_length=64)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
