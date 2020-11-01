from transformers import ElectraForTokenClassification, ElectraTokenizer
from .crf_module import CRF

class ElectroForTokenClassificationCrf(ElectraForTokenClassification):
    def __init__(self,config):
        super(ElectroForTokenClassificationCrf, self).__init__(config=config)
        self.crf = CRF(nb_labels=32, bos_tag_id=101,eos_tag_id=102,pad_tag_id=-100)

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
                                                              return_dict)
        loss1, logits = elector_out[:2]
        loss2 = self.crf(logits, mask=attention_mask, tags=labels)
        return (loss1 + loss2, logits)


if __name__ == '__main__':
    pass
