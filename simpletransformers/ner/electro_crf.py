from transformers import ElectraPreTrainedModel, ElectraModel
import torch.nn as nn
import torch
from .crf_module import CRF

class ElectraForTokenClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cross_entropy_ignore_index=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=cross_entropy_ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

class ElectroForTokenClassificationCrf(ElectraForTokenClassification):
    def __init__(self,config):
        super(ElectroForTokenClassificationCrf, self).__init__(config=config)
        self.crf = CRF(nb_labels=37, bos_tag_id=1,eos_tag_id=2,pad_tag_id=0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cross_entropy_ignore_index=None,
            train_mode=None,
    ):
        elector_out = super(ElectroForTokenClassificationCrf, self).forward(input_ids,
                                                              attention_mask,
                                                              token_type_ids,
                                                              position_ids,
                                                              head_mask,
                                                              inputs_embeds,
                                                              labels,
                                                              output_attentions,
                                                              output_hidden_states,
                                                              return_dict,
                                                              cross_entropy_ignore_index)
        loss1, logits = elector_out[:2]
        loss2 = self.crf(logits, mask=attention_mask, tags=labels)
        if not train_mode:
            sentence_length = input_ids.shape[1]
            scores, logits = self.crf.decode(logits, mask=attention_mask)
            logits = [i + [0]*(sentence_length - len(i)) for i in logits]
            logits = torch.tensor(logits)
        return (loss1 + loss2, logits)


if __name__ == '__main__':
    pass
