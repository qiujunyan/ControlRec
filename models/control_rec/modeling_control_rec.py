from collections import defaultdict

import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
# from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration
)

from utils import cross_entropy_loss, cross_entropy_loss_with_mask


# logger = logging.get_logger(__name__)


# The encoder for input token sequence
class ControlRec(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.name = "ControlRec"

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            slot_ids=None,
            slot_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            reduce_loss=False,
            **kwargs
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = len(input_ids) if input_ids is not None else len(encoder_outputs.last_hidden_state)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
            prompt_attention_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=prompt_attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            prompt_hidden_states = prompt_attention_outputs.last_hidden_state

            # encode slot (user and items)
            assert slot_ids is not None
            slot_attention_mask = kwargs.get("slot_attention_mask", None)
            slot_encoder_outputs = self.encoder(
                input_ids=slot_ids,
                attention_mask=slot_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            slot_hidden_states = slot_encoder_outputs.last_hidden_state

            # prompt_hidden_state和slot_hidden_state，是否需要将T5输出的padding token去除掉后再拼接
            encoder_hidden_states = torch.cat([prompt_hidden_states, slot_hidden_states], 1)

        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state

        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=encoder_hidden_states)

        ##################################################
        ######## Prompt Slot Contrastive Learning ########
        ##################################################
        if kwargs.get("do_pscl", False):
            candidate_slot_ids = kwargs.get("candidate_slot_ids")
            candidate_slot_lens = kwargs.get("candidate_slot_lens")
            candidate_desc_ids = kwargs.get("candidate_desc_ids")
            candidate_desc_lens = kwargs.get("candidate_desc_lens")
            all_negtive_samples = len(candidate_desc_lens)
            assert all_negtive_samples % batch_size == 0
            num_negative_samples = all_negtive_samples // batch_size

            tau = kwargs.get("tau", 1)

            # slot-prompt matching
            # description + negative slots
            candidate_slot_attention_mask = kwargs.get("candidate_slot_attention_mask", None)
            candidate_slot_encoder_outputs = self.encoder(
                input_ids=candidate_slot_ids,
                attention_mask=candidate_slot_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            # shape -> (batch_size * negative_samples, slot_len, model_dim)
            candidate_slot_hidden_states = candidate_slot_encoder_outputs.last_hidden_state
            candidate_slot_cls = candidate_slot_hidden_states[:, 0]
            # shape -> (batch_size, negative_samples, model_dim)
            candidate_slot_cls = candidate_slot_cls.reshape(batch_size, num_negative_samples, -1)
            desc_cls = prompt_hidden_states[:, 0].unsqueeze(1)
            desc_slot_similarity = torch.matmul(desc_cls, candidate_slot_cls.transpose(1, 2))
            desc_slot_similarity = desc_slot_similarity.squeeze(1) / tau

            # slots + negative description
            candidate_desc_attention_mask = kwargs.get("candidate_desc_attention_mask", None)
            candidate_desc_encoder_outputs = self.encoder(
                input_ids=candidate_desc_ids,
                attention_mask=candidate_desc_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            candidate_desc_hidden_states = candidate_desc_encoder_outputs.last_hidden_state
            candidate_desc_cls = candidate_desc_hidden_states[:, 0]
            # shape -> (batch_size, negative_samples, model_dim)
            candidate_desc_cls = candidate_desc_cls.reshape(batch_size, num_negative_samples, -1)
            slot_cls = slot_hidden_states[:, 0].unsqueeze(1)
            slot_desc_similarity = torch.matmul(slot_cls, candidate_desc_cls.transpose(1, 2))
            slot_desc_similarity = slot_desc_similarity.squeeze(1) / tau

            # PSCL loss
            slot_labels = kwargs.get("slot_labels")
            desc_labels = kwargs.get("desc_labels")
            pscl_slot_loss = cross_entropy_loss(slot_labels, slot_desc_similarity, reduction="sum", eps=self.config.eps)
            pscl_desc_loss = cross_entropy_loss(desc_labels, desc_slot_similarity, reduction="sum", eps=self.config.eps)
            return pscl_slot_loss, pscl_desc_loss

        ##################################################
        ################ Auto-Regression #################
        ##################################################
        # 计算拼接后的mask
        if attention_mask is None:
            prompt_attention_mask = torch.ne(input_ids, self.config.pad_token_id)
            slot_attention_mask = torch.ne(slot_ids, self.config.pad_token_id)
            attention_mask = torch.cat([prompt_attention_mask, slot_attention_mask], 1)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/
            # mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            # if reduce_loss:
            #     loss_fct = CrossEntropyLoss(ignore_index=-100)
            # else:
            #     loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = cross_entropy_loss_with_mask(labels, lm_logits, eps=self.config.eps)
            loss = loss.reshape(-1)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        output = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state
        )
        mode = kwargs.get("mode", "train")
        if mode == "valid":
            setattr(output, "last_hidden_state", encoder_hidden_states)
            return output, attention_mask

        ##################################################
        ########## Prompt Contrastive Learning ###########
        ##################################################
        if kwargs.get("do_pcl", False):
            candidate_prompt_ids = kwargs.get("candidate_prompt_ids")
            all_candidate_prompts = len(candidate_prompt_ids)
            assert all_candidate_prompts % batch_size == 0
            num_candidate_prompts = all_candidate_prompts // batch_size

            candidate_prompt_attention_mask = kwargs.get("candidate_prompt_attention_mask")
            candidate_prompt_encoder_output = self.encoder(candidate_prompt_ids,
                                                           attention_mask=candidate_prompt_attention_mask,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)
            prompt_hidden_states = candidate_prompt_encoder_output.last_hidden_state
            slot_hidden_states_expanded = slot_hidden_states.unsqueeze(1).repeat(1, num_candidate_prompts, 1, 1)
            slot_hidden_states_expanded = slot_hidden_states_expanded.reshape(batch_size * num_candidate_prompts,
                                                                              slot_hidden_states.size(1), -1)
            hidden_states_expanded = torch.cat([prompt_hidden_states, slot_hidden_states_expanded], 1)

            # attention mask
            prompt_attention_mask = torch.ne(candidate_prompt_ids, self.config.pad_token_id)
            slot_attention_mask = torch.ne(slot_ids, self.config.pad_token_id)  # (batch_size, seq_len)
            slot_attention_mask = slot_attention_mask.unsqueeze(1).repeat(1, num_candidate_prompts, 1)
            slot_attention_mask = slot_attention_mask.reshape(batch_size * num_candidate_prompts, -1)
            pcl_encoder_attention_mask = torch.cat([prompt_attention_mask, slot_attention_mask], 1)

            encoder_output_pcl = Seq2SeqLMOutput()
            setattr(encoder_output_pcl, "last_hidden_state", hidden_states_expanded)
            generated_output = self.generate(
                encoder_outputs=encoder_output_pcl,
                attention_mask=pcl_encoder_attention_mask,
                return_dict=True,
                return_dict_in_generate=True,
                output_attentions=True,
                output_hidden_states=True
            )

            # pcl generated representation
            # 本机跑这个
            # pcl_decoder_hidden_states = [out[-1] for out in generated_output.decoder_hidden_states]
            # pcl_decoder_hidden_states = torch.cat(pcl_decoder_hidden_states, 1)
            # 服务器跑这个：最后一个token，最后一层
            # TODO
            pcl_decoder_hidden_states = generated_output.decoder_hidden_states[-1][-1]
            generated_seq = generated_output.sequences[:, 1:]
            pcl_mask_decoder = torch.zeros_like(generated_seq, dtype=torch.bool).to(pcl_decoder_hidden_states.device)
            for i, mask in enumerate(generated_seq):
                for j, elem in enumerate(mask):
                    if elem != self.tokenizer.eos_token_id:
                        pcl_mask_decoder[i, j] = True
                    else:
                        break

            # pcl_mask_decoder.shape -> (batch_size * num_candidate, seq_len, 1)
            pcl_mask_decoder = pcl_mask_decoder.unsqueeze(-1)
            # pcl_decoder_hidden_states.shape -> (batch_size * num_candidate, seq_len, embed_size)
            pcl_decoder_hidden_states = pcl_decoder_hidden_states.masked_fill(~pcl_mask_decoder, 0)
            pcl_pooled_hidden_states = pcl_decoder_hidden_states.sum(1) / pcl_mask_decoder.sum(1)
            # pcl_pooled_hidden_states.shape -> (batch_size, num_candidate, embed_size)
            pcl_pooled_hidden_states = pcl_pooled_hidden_states.reshape(batch_size, num_candidate_prompts,
                                                                        -1)

            # target representation
            target_hidden_states = decoder_outputs.last_hidden_state
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ne(decoder_input_ids, self.tokenizer.pad_token_id)
            decoder_attention_mask = decoder_attention_mask.unsqueeze(-1)
            # target_hidden_states.shape -> (batch_size, seq_len, embed_size)
            target_hidden_states = target_hidden_states.masked_fill(~decoder_attention_mask, 0)
            # target_pooled_hidden_states.shape -> (batch_size, 1, embed_size)
            target_pooled_hidden_states = target_hidden_states.sum(1) / decoder_attention_mask.sum(1)
            target_pooled_hidden_states = target_pooled_hidden_states.unsqueeze(1)

            # infoNCE loss
            tau = kwargs.get("tau", 1)
            candidate_prompt_labels = kwargs.get("candidate_prompt_labels")
            pcl_similarity = torch.matmul(target_pooled_hidden_states, pcl_pooled_hidden_states.transpose(1, 2))
            pcl_similarity = pcl_similarity.squeeze(1) / tau
            pcl_loss = cross_entropy_loss(candidate_prompt_labels, pcl_similarity,
                                          reduction="sum", eps=self.config.eps)
            setattr(output, "pcl_loss", pcl_loss)
            setattr(output, "pcl_similarity", pcl_similarity)
            setattr(output, "candidate_prompt_labels", candidate_prompt_labels)
            setattr(output, "target_pooled_hidden_states", target_pooled_hidden_states)
        return output

    def prepare_inputs_for_generation(self,
                                      input_ids=None,
                                      past=None,
                                      attention_mask=None,
                                      encoder_attention_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      **kwargs):

        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "encoder_attention_mask": encoder_attention_mask,
            "use_cache": use_cache,
        }

        return output

    def compute_gen_acc(self, logits, label_ids):
        gen_seq_ids = torch.argmax(logits, -1)
        gen_seq_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in gen_seq_ids]
        mask = torch.eq(label_ids, -100)

        label_ids = torch.max(label_ids, torch.zeros_like(label_ids).to(torch.long))
        golden_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in label_ids]

        gen_hit = torch.eq(gen_seq_ids, label_ids).masked_fill(mask, 0)
        gen_acc = gen_hit.to(torch.float32).sum() / (~mask).to(torch.float32).sum()
        return gen_acc, gen_seq_tokens


class ControlRecPretraining(ControlRec):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')

    def train_step(self, batch, do_pscl=True, do_pcl=True, tau=1, alpha=1):
        device = next(self.parameters()).device
        batch_rec, batch_pscl = batch
        results = defaultdict(float)

        if batch_rec["batch_size"] != 0:
            input_ids = batch_rec['input_ids'].to(device)
            target_ids = batch_rec["target_ids"].to(device)
            slot_ids = batch_rec["slot_ids"].to(device)

            if do_pcl:
                candidate_prompt_ids = batch_rec["prompt_ids"].to(device)
                candidate_prompt_labels = batch_rec["prompt_labels"].to(device)
            else:
                candidate_prompt_ids, candidate_prompt_labels = None, None

            # loss_weights = batch["loss_weights"].to(device)
            output = self(
                input_ids=input_ids,
                labels=target_ids,
                slot_ids=slot_ids,
                return_dict=True,
                do_pcl=do_pcl,
                candidate_prompt_ids=candidate_prompt_ids,
                candidate_prompt_labels=candidate_prompt_labels,
                tau=tau,
            )
            assert 'loss' in output

            lm_mask = target_ids != -100
            lm_mask = lm_mask.float()
            batch_size, label_len = target_ids.size()

            gen_loss = output['loss']
            gen_loss = gen_loss.view(batch_size, label_len) * lm_mask
            # gen_loss.shape -> (batch_size)
            gen_loss = gen_loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

            results['loss'] = gen_loss.mean()
            results["loss_count"] += batch_size
            results['gen'] = gen_loss.mean().detach()
            results['gen_count'] = batch_size

            task_counts = defaultdict(int)
            task_loss = defaultdict(int)
            for _loss, task in zip(gen_loss.detach(), batch_rec['task']):
                task_loss[task] += _loss
                task_counts[task] += 1

            for task in self.losses:
                if task_counts[task] > 0:
                    results[f'{task}'] = task_loss[task]
                    results[f'{task}_count'] = task_counts[task]

            if do_pcl:
                results["pcl_count"] = len(input_ids)
                pcl_loss = output.pcl_loss / results["pcl_count"]
                results["pcl"] = pcl_loss.detach()
                results["loss"] = results["loss"] + alpha * pcl_loss
                results["loss_count"] += results["pcl_count"]

        if do_pscl and batch_pscl["batch_size"] != 0:
            input_ids_pscl = batch_pscl["input_ids"].to(device)
            slot_ids_pscl = batch_pscl["slot_ids"].to(device)
            candidate_slot_ids = batch_pscl["candidate_slot_ids"].to(device)
            candidate_slot_lens = batch_pscl["candidate_slot_lens"].to(device)
            candidate_desc_ids = batch_pscl["candidate_desc_ids"].to(device)
            candidate_desc_lens = batch_pscl["candidate_desc_lens"].to(device)
            slot_labels = batch_pscl["slot_labels"].to(device)
            desc_labels = batch_pscl["desc_labels"].to(device)

            slot_loss, desc_loss = self(input_ids=input_ids_pscl,
                                        attention_mask=None,
                                        slot_ids=slot_ids_pscl,
                                        slot_attention_mask=None,
                                        candidate_slot_ids=candidate_slot_ids,
                                        candidate_slot_lens=candidate_slot_lens,
                                        candidate_desc_ids=candidate_desc_ids,
                                        candidate_desc_lens=candidate_desc_lens,
                                        slot_labels=slot_labels,
                                        desc_labels=desc_labels,
                                        do_pscl=True,
                                        tau=tau)
            results["slot_count"] = batch_pscl["batch_size"]
            results["desc_count"] = batch_pscl["batch_size"]
            slot_loss = slot_loss / results["slot_count"]
            desc_loss = desc_loss / results["desc_count"]
            results["slot"] = slot_loss.detach()
            results["desc"] = desc_loss.detach()

            results["loss"] = results["loss"] + slot_loss + desc_loss
            results["loss_count"] = results["loss_count"] + results["desc_count"] + results["slot_count"]
        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        batch = batch[0]
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)
        slot_ids = batch["slot_ids"].to(device)

        # loss_weights = batch["loss_weights"].to(device)

        output, encoder_attention_mask = self(
            input_ids=input_ids,
            labels=lm_labels,
            slot_ids=slot_ids,
            return_dict=True,
            mode="valid"
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        batch_size, seq_len = lm_labels.size()

        loss = output['loss']

        loss = loss.view(batch_size, seq_len) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = loss.mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}'] = task_loss[task]
                results[f'{task}_count'] = task_counts[task]

        if 'rating' in self.losses:
            generated_output = self.generate(
                encoder_outputs=output,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )

            generated_output_wo_slot = self.generate(input_ids)

            generated_score = self.tokenizer.batch_decode(generated_output, skip_special_tokens=True)
            generated_score_wo_slot = self.tokenizer.batch_decode(generated_output_wo_slot, skip_special_tokens=True)

            results['rating_pred'] = generated_score

        return results

    @torch.no_grad()
    def generate_step(self, batch=None, encoder_outputs=None, attention_mask=None):
        self.eval()
        device = next(self.parameters()).device
        if batch is not None:
            prompt_ids = batch['input_ids'].to(device)
            slot_ids = batch["slot_ids"].to(device)
            input_ids = torch.cat([prompt_ids, slot_ids], -1)
            # input_ids = prompt_ids
            # encoder_attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
            output = self.generate(input_ids=input_ids)

        elif encoder_outputs is not None and attention_mask is not None:
            output = self.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask
            )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents
